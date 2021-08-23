

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
    (3,'uint8',np.uint8),
    (4,'int16',np.int16),
    (5,'uint16',np.uint16),
    (6,'int16le',np.dtype("<i8")),
    (7,'uint16le',np.dtype("<u8")),
    (8, 'int16be', np.dtype(">i8"))
) # TODO add the rest of the datatypes as defined in mrtrix.readthedocs.io

class MIFHeader(LabeledWrapStruct):

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
        # check if file is a mif file or a mih file
        
        # open specified file
        file = open(self.filename,'r+b')
        
        # use fileutils to extract entire header

        fullHeader = read_zt_byte_strings(file)[0].decode("utf-8").split("\n")

        for fields in fullHeader:
            field = fields.split(":")
            if field[0] in self.headerFields:
                fieldName = field[0]
                fieldValue = field[1]
                self.header[fieldName] = []
                self.header[fieldName].append(fieldValue)

        file.close()


    def initDataFile(self):
        
        # open datafile
        if self.datafile is None:
            self.file_ptr = open(self.filename, "r+b")

        else:
            self.file_ptr = open(self.datafile, "r+b")

        self.mmap = mmap.mmap(self.file_ptr.fileno(), length = 0)


    def closeFile(self):
        self.file_ptr.close()
        self.mmap.close()



    
    """        
    ----------------------------------------------------------------------------------------------
       Getters:
    ----------------------------------------------------------------------------------------------
    """
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
                print(item[0].split(split))
                split_header = np.append(split_header,[item[0].split(split)])
                #print(ar)
            if len(split_header) == 1:
                #print(split_header)
                return split_header
            else:
                print(split_header)
                return split_header
                
    def getVoxelLocation(self, absoluteCoordinates):
        """
        Get relative voxel location in the 1D array
        """

        if len(absoluteCoordinates) != len(self.stride):
            print("Absolute coordinates must be the same length as stride")
            return

        voxelLocation = sum([absoluteCoordinates[i]*self.stride[i] for i in range(len(absoluteCoordinates))])
        return voxelLocation

    def convertToByteValue(self, voxelLocation):
        
        try:
            tmp = self.datatype
        except:
            print("Datatype not set\nSetting datatype")
            self.setType()

        try:
            tmp = self.offset
        except:
            print("Offset not set\nSetting offset")
            self.setOffset()

        # dictionary of types in bytes 
        typeSizes = {
            "Bit"       : 1, # Placeholder
            "Int8"      : 1,
            "UInt8"     : 1,
            "Int16"     : 2,
            "UInt16"    : 2,
            "Int16LE"   : 2,
            "UInt16LE"  : 2,
            "Int16BE"   : 2,
            "UInt16BE"  : 2,
            "Int32"     : 4,
            "UInt32"    : 4,
            "Int32LE"   : 4,
            "UInt32LE"  : 4,
            "Int32BE"   : 4,
            "UInt32BE"  : 4,
            "Float32"   : 4,
            "Float32LE" : 4,
            "Float32BE" : 4,
            "Float64"   : 8,
            "Float64LE" : 8,
            "Float64BE" : 8,
            "CFloat32"  : 4,
            "CFloat32LE": 4,
            "CFloat32BE": 4,
            "CFloat64"  : 8,
            "CFloat64LE": 8,
            "CFloat64BE": 8
        }

        byteValue = self.offset + voxelLocation * typeSizes[self.datatype]
        return byteValue


    """        
    ----------------------------------------------------------------------------------------------
       Setters:
    ----------------------------------------------------------------------------------------------
    """
    def setDim(self):
        # store dimensions as [x y z b]
        self.dim = [int(i) for i in self.header['dim'][0].split(",")]

        # set maximum number of bytes
        self.max_byte = 1

        for i in self.dim:
            self.max_byte *= i


    def getTypeSize(self):

        return

    def setLayout(self):
        # store layout variable
        self.layout = [int(i) for i in self.header['layout'][0].split(",")]

    def setOffset(self):
        # checks if its a mih file
        if self.datafile is None:
            self.offset = int((self.header["file"][0].lstrip().split(" ")[1]))

        else:
            print(".mih file detected. No offset")

    def setStride(self):
        # use the minimum values to find out which voxels are which
        # check if dimensions are stored
        try:
            tmp = self.dim[0]
        except:
            print("dim not set\nSetting dim")
            self.setDim()

        try:
            tmp = self.layout[0]
        except:
            print("layout not set\nSetting layout")
            self.setLayout()

        n_dim = len(self.layout)

        # initialise array with size n_dim
        self.stride = [None] * n_dim

        # temporary variable to determine how many types of strides has been passed
        passes = 0

        # need the absolute value of layout to determine which way to traverse first
        abs_layout = list(np.abs(self.layout))
        
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
        if "LE" in self.datatype:
            endianness = "<"
        elif "BE" in self.datatype:
            endianness = ">"

        # find out what datatype the values are stored as
        dtype_mapping = {
            "Int":"i",
            "UInt":"uint",
            "Float":"f",
            "CFloat":"c"
        }

        for keys in dtype_mapping.keys():
            if keys in self.datatype:
                dt = dtype_mapping[keys]

        # get size of datatype in bytes
        size = filter(str.isdigit, self.datatype)
        size = "".join(size)
        size = str(int(size)//8)

        dt = np.dtype(dt+size)
        dt = dt.newbyteorder(endianness)

        self.dtype = dt

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

    def getNextCoordinates(self, previousCoordinate, desiredStride):
        
        move_index = desiredStride.index(min(np.abs(desiredStride)))

        print(move_index)