

from nibabel.fileutils import read_zt_byte_strings
import numpy as np

from .volumeutils import (
    Recoder, native_code, swapped_code, make_dt_codes,
    shape_zoom_affine, array_from_file, seek_tell,
    apply_read_scaling
)


from .arraywriters import (
    make_array_writer, get_slope_inter,
    WriterError, ArrayWriter
)

from .wrapstruct import LabeledWrapStruct

from .spatialimages import (HeaderTypeError, HeaderDataError, SpatialImage)

from .batteryrunners import Report
from .arrayproxy import ArrayProxy

import mmap


default_header = [
    "Filename",
    "dim",
    "vox",
    "layout",
    "datatype",
    "transform",
    "EchoTime",
    "FlipAngle",
    "PhaseEncodingDirection",
    "RepetitionTime",
    "TotalReadoutTime",
    "command_history",
    "comments",
    "dw_scheme",
    "mrtrix_version",
    "offset",
    "file"
]

header_end = "END"

command_history = []

_dtdefs = ( # datatypes for datatype used in header definition 
    (0,'none',np.void),
    (1,'binary',np.void),
    (2,'int8',np.int8),
    (3,'uint8',np.uint8 ),
    (4,'int16',np.int16),
    (5,'uint16',np.uint16),
    (6,'int16le',np.dtype("<i8")),
    (7,'uint16le',np.dtype("<u8")),
    (8, 'int16be', np.dtype(">i8"))
) # TODO add the rest of the datatypes as defined in mrtrix.readthedocs.io

class MIFReader(LabeledWrapStruct):

    """
        Class for MIF/MIH file headers

        ----------------------------------------------------------------------------------------------
        Compulsory arguments:
        ----------------------------------------------------------------------------------------------
        filename        :   Name of mif or mih file as a string
                            eg. "dwi_den.mif", "dwi_den.mih"
        
        ----------------------------------------------------------------------------------------------
        Optional arguments:
        ----------------------------------------------------------------------------------------------
        datafile        :   Name of data file if filename is a mih file
        
        headerFields    :   Header fields in the form of list of strings     
                            If no headers are specified, the default header fields are used                       
                            eg. See default_header

        ----------------------------------------------------------------------------------------------
        Variables:
        ----------------------------------------------------------------------------------------------
        filename    :   name of file as a string
                        eg. "dwi_den.mif"
        file_ptr    :   initialised datafile pointer
        datafile    :   name of datafile associated with .mih file
                        None if file is a mif file
        mmap        :   memory mapping of fileptr
        header      :   Header variables and values based on header fields specified
        dtype       :   np.dtype used to reconstruct value
        layout      :   array of strings with the data layout
        stride      :   array of integers with the stride for each axis
        sign        :   array made of {-1,+1} derived from layout
                        used for calculation of the next coordinate in the 3-D or 4D volumes
        


    """
    def __init__(
        self,
        filename,
        datafile = None,
        headerFields = None
        ):

        self.filename = filename

        if ".mif" in filename:
            self.datafile = None
        elif ".mih" in filename:
            self.datafile = datafile

        self.header = {}
        
        # allow users to add header fields that they want
        if headerFields is None:
            self.headerFields = default_header

        else:
            self.headerFields = headerFields
    
#---------------------------------------------------#
#      Helper functions to complete header          #
#---------------------------------------------------#

    def setDim(self):
        # store dimensions as [x y z b]
        self.dim = [int(i) for i in self.header['dim'][0].split(",")]

        # set maximum number of bytes
        self.max_byte = 1

        for i in self.dim:
            self.max_byte *= 1


    def setLayout(self):

        self.layout = [i for i in self.header['layout'][0].lstrip().split(",")]

    def setSign(self):
        self. sign = []
        
        for i in self.layout:
            if i[0] == "+":
                self.sign.append(1)

            elif i[0] == "-":
                self.sign.append(-1)

    def setOffset(self):

        # if its a mif file theres an offset
        if self.datafile is None:
            self.offset = int((self.header["file"][0].lstrip().split(" ")[1]))

        # if its a mih file the data file has no offset
        else:
            self.offset = 0

    def setStride(self):
        # use the minimum values to find out which voxels are which
        # check if dimensions are stored
        try:
            tmp = self.dim[0]
        except:
            self.setDim()

        try:
            tmp = self.layout[0]
        except:
            self.setLayout()

        n_dim = len(self.layout)

        # initialise array with size n_dim
        self.stride = [None] * n_dim

        # temporary variable to determine how many types of strides has been passed
        passes = 0

        # need the absolute value of layout to determine which way to traverse first
        abs_layout = [int(i) for i in self.layout]
        abs_layout = list(np.abs(abs_layout))
        
        currentMultiplier = 1

        while passes < n_dim:
            min_index = abs_layout.index(min(abs_layout))

            self.stride[min_index] = currentMultiplier
            currentMultiplier *= self.dim[min_index]

            abs_layout[min_index] = n_dim + 1

            passes += 1

    def setType(self):
        self.datatype = self.getHeaderValue("datatype")[0].lstrip()
        # assume endianness is LE unless stated
        endianness = "<"

        # get the endianness
        if "le" in self.datatype.lower():
            endianness = "<"
        elif "be" in self.datatype.lower():
            endianness = ">"

        # find out what datatype the values are stored as
        dtype_mapping = {
            "Int":"i",
            "UInt":"uint",
            "Float":"f",
            "CFloat":"c"
        }

        for keys in dtype_mapping.keys():
            if keys.lower() in self.datatype.lower():
                dt = dtype_mapping[keys]

        # get size of datatype in bytes
        size = filter(str.isdigit, self.datatype)
        size = "".join(size)
        size = str(int(size)//8)

        dt = np.dtype(dt+size)
        dt = dt.newbyteorder(endianness)

        self.dtype = dt

    def getHeaderValue(self, key, split = None, type = None):
        """
        returns value of a header field

        ----------------------------------------------------------------------------------------------
        split options
        ----------------------------------------------------------------------------------------------
        split = None
        return value of specified header field as list of Python strings

        split = ","
        splits header field values

        ----------------------------------------------------------------------------------------------
        type options
        ----------------------------------------------------------------------------------------------
        type = None
        return as Python strings

        type = "i8"
        attempts to cast according to the numpy dtypes
        #TODO type casting
        """

        try:
            value = self.header[key]
        except:
            print("MIF/MIH HEADER ERROR: Key not found")
            return

        if split == None and type == None:

            return value

        
        split_header = np.array([])

        if split is not None:
        
            # iterate over header values
            for item in value:
                split_header = np.append(split_header,[item[0].split(split)])
                #print(ar)
            if len(split_header) == 1:
                #print(split_header)
                return split_header
            else:
                return split_header

#---------------------------------------------------#
#                 Complete header                   #
#---------------------------------------------------#
    

    def findHeader(self, header2file = None):

        """
            Method to find header information for a mif/mih file
            Uses instance variable headerField

            #TODO implement helper functions to clean up

            ----------------------------------------------------------------------------------------------
            headerField options:
            ----------------------------------------------------------------------------------------------
            headerField = [field1, field2, ..., fieldN]
            Searches mif/mih file until header_end = "END" is found
            If header field is found in headerField, the header field is added to dictionary or appended if it already exists

            #TODO indicate if a field is not found

            headerField = [None]
            Searches mif/mih file under header_end = "END" is found
            Adds all header fields and values to the dictionary and appends if it already exists

            ----------------------------------------------------------------------------------------------
            header2file options:
            ----------------------------------------------------------------------------------------------
            Optional argument

            header2file = None
            Nothing from the header is saved to an external file

            header2file = [[headerField1, [file1, file2, ..., fileN]], [headerField2, [file2, file3, ...]]]
            
            #TODO implement outputting specified headers to file
        """
        
        # open specified file
        file = open(self.filename,'rb')
        
        # use fileutils to extract entire header

        fullHeader = read_zt_byte_strings(file)[0].decode("utf-8").split("\n")
        for fields in fullHeader:
            field = fields.split(":")
            if field[0] in self.headerFields:
                fieldName = field[0]
                fieldValue = field[1]
                self.header[fieldName] = []
                self.header[fieldName].append(fieldValue)


        self.setLayout()
        self.setStride()
        self.initDataFile()
        self.setType()
        self.setOffset()
        self.setSign()

        file.close()

#---------------------------------------------------#
#                   Datafile stuff                  #
#---------------------------------------------------#

    def initDataFile(self):

        # open datafile
        if self.datafile is None:
            self.file_ptr = open(self.filename, "r+b")
        
        else:
            self.file_ptr = open(self.datafile, "r+b")

        self.mmap = mmap.mmap(self.file_ptr.fileno(), length = 0)


#---------------------------------------------------#
#                   Get voxels                      #
#---------------------------------------------------#

    def getVoxel(self, byteLocation):    
        # gets voxel value based on bytelocation given
        # uses numpy datatypes to determine how many bytes to read

        self.mmap.seek(byteLocation)
        size = filter(str.isdigit, self.datatype)
        size = "".join(size)
        size = (int(size)//8)
        value = self.mmap.read(size)
        value = np.frombuffer(value, self.dtype)

        return value

#---------------------------------------------------#
#                   Get intensity value             #
#---------------------------------------------------#

    def getVoxelIntensity(self, coordinate):

        if len(coordinate) != len(self.dim):
            print("Incorrect dimensions")
            return

        if sum([i<0 for i in coordinate]) > 0:
            print("Invalid coordinate")

            return


        byteval = self.getByte(coordinate)


        intensity = self.getVoxel(byteval)

        return intensity


#---------------------------------------------------#
#         Get next coordinates based on layout      #
#---------------------------------------------------#

    def getNextCoordinates(self, previousCoordinate, layout = None, sign = None):
        

        if layout is None:
            desiredLayout_tmp = list(np.abs([int(i) for i in self.layout]))

        else:
            desiredLayout_tmp = list(np.abs([int(i) for i in layout]))


        if sign is None:
            desiredSign = self.sign

        else:
            desiredSign = sign



        for i in range(0,len(desiredLayout_tmp)):
            # use the layout to determine which direction to move in
            idx = desiredLayout_tmp.index(min(np.abs(desiredLayout_tmp)))
            desiredLayout_tmp[idx] = 999
            
            nextPossibleCoordinate = previousCoordinate[idx] + 1*desiredSign[idx]

            # print("idx = %d, val = %d"%(idx,nextPossibleCoordinate))
            if nextPossibleCoordinate < self.dim[idx] and nextPossibleCoordinate >= 0:
                nextCoordinate = previousCoordinate
                nextCoordinate[idx] += 1*desiredSign[idx]
                break

            if nextPossibleCoordinate >= self.dim[idx]:
                nextCoordinate = previousCoordinate
                nextCoordinate[idx] = self.getStartingCoordinates(desiredSign)[idx]
            
            if nextPossibleCoordinate < 0:
                nextCoordinate = previousCoordinate
                nextCoordinate[idx] = self.dim[idx]-1


        return nextCoordinate

#---------------------------------------------------#
#      Get origin based off starting point          #
#---------------------------------------------------#

    def getStartingCoordinates(self, sign = None):
    
        startingCoordinates = []

        if sign is None:
            desiredSign = self.sign

        else:
            desiredSign = sign

        i = 0
        while i < len(desiredSign):
            if desiredSign[i] < 0:
                startingCoordinates.append(self.dim[i]-1)

            elif desiredSign[i] > 0:
                startingCoordinates.append(0)

            i += 1

        return startingCoordinates

    def getByte(self, coordinate):

        absDifferenceFromOrigin = np.abs([coordinate[i] - self.getStartingCoordinates()[i] for i in range(len(coordinate))])
        size = int(''.join(filter(str.isdigit, self.datatype)))
        byte = self.offset + sum(self.stride*absDifferenceFromOrigin)*(size//8)

        return byte

    def changeLayout(self, newLayout, newSign, filename):

        if newSign is None:
            newSign = self.sign

        self.writeHeader(newLayout, newSign, filename)

        f = open(filename, "ab")


        # to keep track of the number of voxels
        totalVoxels = 1

        # how many voxels are there in the file
        for i in self.dim:
            totalVoxels *= i

        coordinates = self.getStartingCoordinates(newSign)
        print("Starting from coordinates %s"%coordinates)

        currentVal = 0
        currentByte = self.getByte(coordinates)
        while currentVal < totalVoxels:
            voxelValue = self.getVoxel(currentByte)

            print("Writing %s, %d from byte %d from old file at position %d in new file"%(coordinates,voxelValue,currentByte, f.tell()))
            voxelValue = voxelValue.astype(self.dtype).tobytes()

            f.write(voxelValue)


            coordinates = self.getNextCoordinates(coordinates, newLayout, newSign)

            currentByte = self.getByte(coordinates)

            currentVal += 1

        f.close()

        return

    def writeHeader(self, newLayout, newSign, filename):
        f = open(filename, "w")

        f.write("mrtrix image\n")

        for key in self.header.keys():
            
            items = self.header[key]

            f.write("%s:"%key)

            for item in items:

                if key == "layout":
                    layout = self.getOutputLayout(newLayout, newSign)

                    f.write(layout)

                elif key == "file":
                    offset = self.getOutputOffset(f)
                    f.write(offset)

                else:
                    f.write("%s"%("".join(item)))

                f.write("\n")

        return


    def getOutputLayout(self, newLayout,newSign):

        layout = " "

        i = 0

        while i < len(newLayout):
            if newSign[i] > 0:
                layout += ("+%s,"%newLayout[i])

            elif newSign[i] < 0:
                layout += (("-%s,"%newLayout[i]))

            i += 1

        layout = layout[:-1]
                
        print(layout)
        return layout

    def getOutputOffset(self, fpointer):

        offset = fpointer.tell() + 8

        
        offset = offset + np.mod((4-np.mod(offset,4)),4)
        
        s = (". %d\nEND\n                      "%offset)
        print(
            
        )
        s = s[0:(offset-fpointer.tell())-3]
        

        return s



