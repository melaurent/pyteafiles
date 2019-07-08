# Copyright (C) 2011 discretelogics
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import struct
import uuid
from io import BytesIO
from collections import namedtuple
import ctypes
from ctypes import Structure
from ctypes import c_int8, c_int16, c_int32, c_int64, c_uint8, c_uint16, c_uint32, c_uint64, c_float, c_double
from datetime import datetime

class TeaFile:
    '''
    Create, write and read or just inspect a file in the TeaFile format.

    1. **create** and **write** a teafile holding Time/Price/Volume
    items.

    Note: Arguments of tf.write show up in intellisense with their names "Time", "Price" and "Volume".

    2. **read** a teafile. TeaFiles are self describing  a filename is sufficient, - we might have no clue what is inside the file, due to
    TeaFiles
    Since the item structure is described in the file, we can always open the data items in the file.
    We can even do so on many platforms and with many applications, like from R on Linux, Mac OS or Windows,
    or using proprietary C++ or C# code.

    3. **describe**: See the `description` property about accessing the values passed to create. As a teaser, lets access
    the content description and namevalues collection for the file above:
    '''

    #pylint:disable-msg=R0902,W0212

    # the factory methods at the module level, `create`, `openread` and `openwrite` should be used to create instances of this class.
    def __init__(self, filename):
        self.decimals = -1

        self._filename = filename   # we need the filename for the call to getsize in itemareaend()
        self.file = None

        self._description = None

        self.itemareastart = None
        self._itemareaend = None
        self.itemsize = None

    @staticmethod
    def create(filename, datatype, contentdescription=None, namevalues=None):
        '''
        creates a new file and writes its header based on the description passed.
        leaves the file open, such that items can be added immediately. the caller must close the
        file finally.

        args:
            * **filename**:   The filename, that will internally be passed to io.open, so the same rules apply.
            * **datatype**: The type of the ctype Structure object
            * **contentdescription**: A teafile can store one contentdescription, a string that describes what the
              contents in the file is about. examples: "Weather NYC", "Network load", "ACME stock".
              Applications can use this string as the "title" of the time series, for instance in a chart.
            * **namevalues**: A collection of name-value pairs used to store descriptions about the file.
              Often additional properties, like the "data provider", "feed", "feed id", "ticker".
              By convention, the name "decimals" is used to store an integer describing how many
              numbers of decimals to be used to format floating point values. This api for instance makes
              use of this convention. Besides formatting, an application might also treat this number
              as the accuracy of floating point values.

        note that itemcount is still accessible, even after the file is closed.
        '''
        tf = TeaFile(filename)

        # setup description
        tf._description = d = TeaFileDescription()
        id_ = ItemDescription.create(datatype)
        d.itemdescription = id_
        d.contentdescription = contentdescription
        d.namevalues = namevalues
        d.timescale = TimeScale.java()

        # open file and write header

        tf.file = open(filename, "wb")
        hm = _HeaderManager()
        fio = _FileIO(tf.file)
        fw = _FormattedWriter(fio)
        wc = hm.writeheader(fw, tf._description)
        tf.itemareastart = wc.itemareastart
        tf._itemareaend = wc.itemareaend
        tf.itemsize = id_.itemsize

        tf.flush()
        return tf

    @staticmethod
    def openread(filename):
        '''
        Open a TeaFile for read only.
        '''
        tf = TeaFile._open(filename, "rb")
        tf.write = None

        return tf

    @staticmethod
    def openwrite(filename):
        '''
        Open a TeaFile for read and write.

        The file returned will have its *filepointer set to the end of the file*, as this function
        calls seekend() before returning the TeaFile instance.
        '''
        tf = TeaFile._open(filename, "r+b")
        tf.seekend()
        return tf

    @staticmethod
    def _open(filename, mode):
        ''' internal open method, used by openread and openwrite '''
        tf = TeaFile(filename)
        tf.file = open(filename, mode)
        fio = _FileIO(tf.file)
        fr = _FormattedReader(fio)
        hm = _HeaderManager()
        rc = hm.readheader(fr)
        tf._description = rc.description
        id_ = tf._description.itemdescription
        if id_:
            tf.itemsize = id_.itemsize
            tf.datatype = id_.datatype
        tf.itemareastart = rc.itemareastart
        tf._itemareaend = rc.itemareaend

        nvs = tf._description.namevalues
        if nvs and nvs.get("decimals"):
            tf.decimals = nvs["decimals"]
        return tf

    def read(self):
        el = self.datatype()
        self.file.readinto(el)
        return el

    def write(self, item):
        self.file.write(item)

    def flush(self):
        '''
        Flush buffered bytes to disk.

        When items are written via write, they do not land directly in the file, but are buffered in memory. flush
        persists them on disk. Since the number of items in a TeaFile is computed from the size of the file, the
        `itemcount` property is accuraty only after items have been flushed.
        '''

        self.file.flush()

    def seekitem(self, itemindex):
        '''
        Sets the file pointer to the item at index `temindex`.
        '''
        if self.write is None:
            raise RuntimeError("seeking in write mode not supported")
        self.file.seek(self.itemareastart + itemindex * self.itemsize)

    def seekend(self):
        '''
        Sets the file pointer past the last item.
        '''
        self.file.seek(0, 2)    # SEEK_END

    def items(self, start=0, end=None):
        '''
        Returns an iterator over the items in the file allowing start and end to be passed as item index.
        Calling this method will modify the filepointer.

        Optional, the range of the iterator can be returned
        '''

        if self.write is None:
            raise RuntimeError("reading in write mode not supported")

        self.seekitem(start)
        if not end:
            end = self.itemcount
        end = min(end, self.itemcount)
        current = start
        while current < end:
            yield self.read()
            current += 1

    @property
    def itemcount(self):
        ''' The number of items in the file. '''
        return self._getitemareasize() / self.itemsize

    def _getitemareaend(self):
        ''' the end of the item area, as an integer '''
        import os
        if self._itemareaend:
            return self._itemareaend
        return os.path.getsize(self._filename)

    def _getitemareasize(self):
        ''' the item area size, as an integer '''
        return self._getitemareaend() - self.itemareastart

    def close(self):
        '''
        Closes the file.

        TeaFile implements the context manager protocol and using this protocol is prefered, so manually closing the file
        should be required primarily in interactive mode.
        '''
        self.file.close()

    # context manager protocol
    def __enter__(self):
        return self

    def __exit__(self, type_, value, tb):
        self.close()

    # information about the file and its contents
    @property
    def description(self):
        '''
        Returns the description of the file.

        TeaFile describe the structure of its items and annotations about its content in their header. This
        property returns this description which in turn (optionally) holds
        * the itemdescription describing field names, types and offsets
        * a contentdescription describing the content
        * a namevalue collection holding name-value pairs and
        * a timescale describing how time stored as numbers shall be interpreted as time.::
        '''
        return self._description

    def __repr__(self):
        return  "TeaFile('{}') {} items".format(self._filename, self.itemcount)

    @staticmethod
    def printitems(filename, maxnumberofitems=10):
        '''
        Prints all items in the file. By default at most 10 items are printed.
        '''
        with TeaFile.openread(filename) as tf:
            from itertools import islice
            print(list(islice(tf.items(), maxnumberofitems)))
            if tf.itemcount > maxnumberofitems:
                print ("{} of {} items".format(maxnumberofitems, tf.itemcount))

    @staticmethod
    def printsnapshot(filename):
        '''
        Prints a snapshot of an existing file, that is its complete description and the first 5 items.
        '''
        with TeaFile.openread(filename) as tf:
            print(tf)
            print("")
            print(tf.description)
            print("Items")
            for item in tf.items(0, 5):
                print(item)


class _ValueKind:   # pylint: disable-msg=R0903
    ''' enumeration type, describing the type of a value inside a name-value pair '''
    Invalid, Int32, Double, Text, Uuid, Uint64 = [0, 1, 2, 3, 4, 5]


def _getnamevaluekind(value):
    ''' returns the `_ValueKind' based on the for the passed `value` '''
    if isinstance(value, int):
        return _ValueKind.Int32
    if isinstance(value, float):
        return _ValueKind.Double
    if isinstance(value, basestring):
        return _ValueKind.Text
    if isinstance(value, uuid):
        return _ValueKind.Uuid
    raise ValueError("Invalid type inside NameValue")


class TimeScale:
    '''
    The TeaFile format is time format agnostic. Times in such file can be integral or float values
    counting seconds, milliseconds from an epoch like 0001-01-01 or 1970-01-01. The epoch together
    with the tick size define the `time scale` modeled by this class. These values are stored in the file.

    In order to support many platforms, the epoch value of 1970-01-01 and a tick size of Milliseconds is recommended.
    Moreover, APIs for TeaFiles should primarily support this time scale before any other, to allow exchange
    between applications and operating systems. In this spirit, the clockwise module in this package uses this
    1970 / millisecond time scale.
    '''
    def __init__(self, epoch, ticksperday):
        self._epoch = epoch
        self._ticksperday = ticksperday

    @staticmethod
    def java():
        '''
        Returns a TimeScale instance with the epoch 1970-01-01 and millisecond resolution.
        This time scale is that used by Java, so we call this the Java TimeScale.
        '''
        return TimeScale(719162, 86400000)

    @property
    def wellknownname(self):
        ''' Returns 'Java' if epoch == 719162 (1970-01-01) and ticksperday == 86400 * 1000.
            Returns 'Net' if epoch == 0 (0001-01-01) and ticksperday == 86400 * 1000 * 1000 * 10.
            Returns None otherwise.
        '''
        if self._epoch == 719162 and self._ticksperday == 86400000:
            return "Java"
        if self._epoch == 0 and self._ticksperday == 864000000000:
            return "Net"
        return None

    def __repr__(self):
        s = "Epoch:         {:>8}\nTicks per Day: {:>8}\n" \
                .format(self._epoch, self._ticksperday)
        wnn = self.wellknownname
        if wnn:
            s += "Wellknown Scale:{:>7}\n".format(wnn)
        return s


class _FileIO:
    ''' FileIO provides the ability to read int32, int64, double and byte lists from a file '''

    def __init__(self, iofile):
        self.file = iofile

    # read
    def readint32(self):
        ''' read a 32bit signed integer from the file '''
        bytes_ = self.file.read(4)
        value = struct.unpack("i", bytes_)[0]
        return value

    def readint64(self):
        ''' read a 64bit signed integer from the file '''
        bytes_ = self.file.read(8)
        value = struct.unpack("q", bytes_)[0]
        return value

    def readdouble(self):
        ''' read a double from the file '''
        bytes_ = self.file.read(8)
        value = struct.unpack("d", bytes_)[0]
        return value

    def readbytes(self, n):
        ''' read `n` bytes from the file '''
        return self.file.read(n)

    # write
    def writeint32(self, value):
        ''' write a 32bit signed integer to the file '''
        bytes_ = struct.pack("i", value)
        assert len(bytes_) == 4
        self.file.write(bytes_)

    def writeint64(self, value):
        ''' write a 64bit signed integer to the file '''
        bytes_ = struct.pack("q", value)
        assert len(bytes_) == 8
        self.file.write(bytes_)

    def writedouble(self, value):
        ''' write a double to the file '''
        bytes_ = struct.pack("d", value)
        assert len(bytes_) == 8
        self.file.write(bytes_)

    def writebytes(self, bytes_):
        ''' write the list of byte to the file '''
        self.file.write(bytes_)

    # position
    def skipbytes(self, bytestoskip):
        ''' skip `bytestoskip` in the file. increments the file pointer '''
        self.file.read(bytestoskip)

    def position(self):
        ''' returns the file pointer '''
        return self.file.tell()


class _FormattedReader:
    ''' Provides formatted reading of a `_FileIO` instance.'''

    def __init__(self, fio):
        self.fio = fio

    def readint32(self):
        ''' read int32 '''
        return self.fio.readint32()

    def readint64(self):
        ''' read int64 '''
        return self.fio.readint64()

    def readdouble(self):
        ''' read double '''
        return self.fio.readdouble()

    def readbytes_lengthprefixed(self):
        ''' read bytes, length prefixed '''
        n = self.readint32()
        return self.fio.readbytes(n)

    def readtext(self):
        ''' read a unicode string in utf8 encoding '''
        return self.readbytes_lengthprefixed().decode("utf8")

    def readuuid(self):
        ''' read a uuid '''
        bytes16 = self.fio.readbytes(16)
        return uuid.UUID(bytes=bytes16)

    def skipbytes(self, bytestoskip):
        ''' skip `bytestoskip` bytes '''
        self.fio.skipbytes(bytestoskip)

    def position(self):
        ''' returns the position of the filepointer '''
        return self.fio.position()

    def readnamevalue(self):
        ''' returns a dictionary holding a single name : value pair '''
        name = self.readtext()
        kind = self.readint32()
        if kind == _ValueKind.Int32:
            value = self.readint32()
        elif kind == _ValueKind.Double:
            value = self.readdouble()
        elif kind == _ValueKind.Text:
            value = self.readtext()
        elif kind == _ValueKind.Uuid:
            value = self.readuuid()
        return {name: value}


class _FormattedWriter:

    def __init__(self, fio):
        self.fio = fio

    def writeint32(self, int32value):
        ''' write an int32 value '''
        self.fio.writeint32(int32value)

    def writeint64(self, int64value):
        ''' write an int64 value '''
        self.fio.writeint64(int64value)

    def writedouble(self, doublevalue):
        ''' write a double value '''
        self.fio.writedouble(doublevalue)

    def writebytes_lengthprefixed(self, bytes_):
        ''' writes the `bytes` prefixed with their length '''
        self.writeint32(len(bytes_))
        self.fio.writebytes(bytes_)

    def writeraw(self, bytes_):
        ''' write `bytes` without length prefixing them '''
        self.fio.writebytes(bytes_)

    def writetext(self, text):
        ''' write `text` in UTF8 encoding '''
        self.writebytes_lengthprefixed(text.encode("utf8"))   # todo: is this encoding right?

    def writeuuid(self, uuidvalue):
        ''' Not implemented yet. writes `uuidvalue` into the file. '''
        raise Exception("cannot write uuid, feature not yet implemented. uuid={}{}".format(self, uuidvalue))
        #bytes16 = self.fio.writebytes(16)
        #return uuid.UUID(bytes=bytes16)

    def skipbytes(self, bytestoskip):
        ''' skip `bytestoskip` '''
        self.fio.skipbytes(bytestoskip)

    def position(self):
        ''' return the position (the file pointer '''
        return self.fio.position()

    def writenamevalue(self, key, value):
        ''' write a name/value pair '''
        kind = _getnamevaluekind(value)
        self.writetext(key)
        self.writeint32(kind)
        if kind == _ValueKind.Int32:
            self.writeint32(value)
        elif kind == _ValueKind.Double:
            self.writedouble(value)
        elif kind == _ValueKind.Text:
            self.writetext(value)
        elif kind == _ValueKind.Uuid:
            self.writeuuid(value)


# descriptions
class TeaFileDescription:
    '''
    Holds the description of a time series. Its attributes are the
        itemdescription, describing the item's fields and layout
        contentdescription, a simple string describing what the time series is about
        namevalues, a collection of name-value pairs holding int32,double,text or uuid values and the
        timescale, describing the format of times inside the file
    '''
    #pylint: disable-msg=R0903

    def __init__(self):
        self.itemdescription = None
        self.contentdescription = None
        self.namevalues = None
        self.timescale = None

    def __repr__(self):
        return "ItemDescription\n{}" \
               "\n\nContentDescription\n{}" \
               "\n\nNameValues\n{}" \
               "\n\nTimeScale\n{}" \
                .format(self.itemdescription, \
                    self.contentdescription, \
                    self.namevalues, \
                    self.timescale)


class ItemDescription:
    '''
    The item description describes the item type.
    Each teafile is a homogenous collection of items and an instance of this class describes
    the fields of this item, that is

        the name of each field
        the field's offset inside the item
        its type.
    '''
    def __init__(self):
        self.itemsize = 0
        self.itemname = ""
        self.fields = []
        self.fieldnames = None
        self.datatype = None

    def __repr__(self):
        from pprint import pformat
        return "Name:\t{}\nSize:\t{}\nFields:\n{}" \
            .format(self.itemname, self.itemsize, pformat(self.fields))

    @staticmethod
    def create(datatype):
        '''
        Creates an ItemDescription instance to be used for the creation of a new TeaFile.

        Infer the item description from the given data type which must be a Structure from ctypes
        '''

        id_ = ItemDescription()
        try:
            if not issubclass(datatype, Structure):
                raise RuntimeError("TeaFile expects a ctype Structure")
        except:
            raise RuntimeError("TeaFile expects a ctype Structure")

        # create Fields
        id_.itemsize = ctypes.sizeof(datatype)
        id_.itemname = datatype.__name__
        i = 0
        for field in datatype._fields_:
            f = Field()
            f.name = field[0]
            df = getattr(datatype, f.name)
            f.offset = df.offset
            f.index = i
            f.fieldtype = FieldType.typetofieldtype(field[1])
            i += 1
            id_.fields.append(f)

        return id_

    def getfieldbyoffset(self, offset):
        ''' Returns a field given its offset '''
        for f in self.fields:
            if f.offset == offset:
                return f
        print("field not found at offset{0}".format(offset))
        raise RuntimeError()

    def setupfromfields(self):
        structfields = []
        for f in self.fields:
            structfields.append((f.name, FieldType.fieldtypetotype(f.fieldtype)))
        datatype = type(str(self.itemname), (Structure,), {
            "_pack_": 0,
            "_fields_": structfields,
        })
        self.datatype = datatype

    @staticmethod
    def _getsafename(name):
        ''' convert item or field name to a name valid for namedtuple '''
        validchars = '_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        return ''.join(c for c in name if c in validchars)


class FieldType:
    ''' An enumeration of field types and utility functions related to. '''

    _types = [None, c_int8, c_int16, c_int32, c_int64, c_uint8, c_uint16, c_uint32, c_uint64, c_float, c_double]
    _typeNames = [None, "Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32", "UInt64", "Float", "Double"]

    @staticmethod
    def typetofieldtype(typ):
        if typ == c_int8:
            return 1
        if typ == c_int16:
            return 2
        if typ == c_int32:
            return 3
        if typ == c_int64:
            return 4
        if typ == c_uint8:
            return 5
        if typ == c_uint16:
            return 6
        if typ == c_uint32:
            return 7
        if typ == c_uint64:
            return 8
        if typ == c_float:
            return 9
        if typ == c_double:
            return 10
        raise RuntimeError("Unknown structure's field type")

    @staticmethod
    def fieldtypetotype(fieldtype):
        return FieldType._types[fieldtype]

    @staticmethod
    def getname(fieldtype):
        return FieldType._typeNames[fieldtype]

class Field:
    def __init__(self):
        self.name = ""
        self.offset = None
        self.fieldtype = None
        self.istime = False
        self.iseventtime = False
        self.index = None

    def getvalue(self, item):
        ''' Given a field and an item, returns the value of this field.

        If the field is a time field, the value is packed into a `Time`, unless
        configured otherwise by setting `use_time_decoration` to False.
        '''
        value = item[self.index]
        if self.istime:
            value = datetime.fromtimestamp(value)
            # TODO transform timestamp to the definied precision
        return value

    def decoratetime(self, item):
        value = item[self.index]
        if isinstance(value, datetime):
            value = value.timestamp()
            # TODO transform timestamp to the definied precision
        return value

    def __repr__(self):
        return "{:10}   Type:{:>7}   Offset:{:>2}   IsTime:{}   IsEventTime:{}".format(self.name, FieldType.getname(self.fieldtype), self.offset, int(self.istime), int(self.iseventtime))


# context, section formatters
class _ReadContext:
    ''' context used for header reading '''
    #pylint:disable-msg=R0903
    def __init__(self, formattedreader):
        self.reader = formattedreader
        self.description = TeaFileDescription()
        self.itemareastart = None
        self.itemareaend = None
        self.sectioncount = None


class _WriteContext:
    ''' context used for header writing '''
    def __init__(self, formattedwriter):
        self.writer = formattedwriter
        self.itemareastart = None
        self.itemareaend = None
        self.sectioncount = None
        self.description = None


class _ItemSectionFormatter:
    ''' reads and writes the itemdescription into / from the file '''
    id = 10

    def read(self, rc):
        ''' read the section '''
        id_ = ItemDescription()
        r = rc.reader
        id_.itemsize = r.readint32()
        id_.itemname = r.readtext()
        fieldcount = r.readint32()
        i = 0
        for _ in range(fieldcount):
            f = Field()
            f.index = i
            f.fieldtype = r.readint32()
            f.offset = r.readint32()
            f.name = r.readtext()
            id_.fields.append(f)
            i += 1
        id_.setupfromfields()
        rc.description.itemdescription = id_

    def write(self, wc):
        ''' writes the section '''
        id_ = wc.description.itemdescription
        w = wc.writer
        w.writeint32(id_.itemsize)
        w.writetext(id_.itemname)
        w.writeint32(len(id_.fields))
        for f in id_.fields:
            w.writeint32(f.fieldtype)
            w.writeint32(f.offset)
            w.writetext(f.name)


class _ContentSectionFormatter:
    ''' reads and writes the contentdescription into / from the file '''
    id = 0x80

    def read(self, rc):
        ''' read the section '''
        r = rc.reader
        rc.description.contentdescription = r.readtext()

    def write(self, wc):
        ''' writes the section '''
        cd = wc.description.contentdescription
        if cd:
            wc.writer.writetext(cd)


class _NameValueSectionFormatter:
    ''' reads and writes the namevalue-description into / from the file '''
    id = 0x81

    def read(self, rc):
        ''' read the section '''
        r = rc.reader
        n = r.readint32()
        if n == 0:
            return
        nvc = {}
        while n > 0:
            nv = r.readnamevalue()
            nvc.update(nv)
            n = n - 1
        rc.description.namevalues = nvc

    def write(self, wc):
        ''' writes the section '''
        nvs = wc.description.namevalues
        if not nvs:
            return
        w = wc.writer
        w.writeint32(len(nvs))
        for key, value in nvs.items():
            w.writenamevalue(key, value)


class _TimeSectionFormatter:
    ''' reads and writes the timescale and description of time fields into / from the file '''
    id = 0x40

    def read(self, rc):
        ''' read the section '''
        r = rc.reader

        # time scale
        epoch = r.readint64()
        ticksperday = r.readint64()
        rc.description.timescale = TimeScale(epoch, ticksperday)

        # time fields
        timefieldcount = r.readint32()
        if timefieldcount == 0:
            return

        id_ = rc.description.itemdescription
        isfirsttimefield = True
        for _ in range(timefieldcount):
            o = r.readint32()
            f = id_.getfieldbyoffset(o)
            f.istime = True
            f.iseventtime = isfirsttimefield
            isfirsttimefield = False

    def write(self, wc):
        ''' writes the section '''
        w = wc.writer
        # this api restricts time formats to JavaTime
        # in addition, the first field named "time" is considered the EventTime
        w.writeint64(719162)        # days between 0001-01-01 and 1970-01-01
        w.writeint64(86400 * 1000)  # millisecond resolution
        id_ = wc.description.itemdescription
        timefields = [f for f in id_.fields if f.name.lower() == "time"]
        w.writeint32(len(timefields))   # will be 0 or 1
        for f in timefields:
            w.writeint32(f.offset)


class _HeaderManager:
    ''' reads and writes the file header, delegating the formatting of sections to the sectionformatters. '''
    def __init__(self):
        self.sectionformatters = ([
            _ItemSectionFormatter(),
            _ContentSectionFormatter(),
            _NameValueSectionFormatter(),
            _TimeSectionFormatter()])

    def getformatter(self, id_):
        ''' the a formatter given its id '''
        for f in self.sectionformatters:
            if f.id == id_:
                return f
        raise RuntimeError()

    def readheader(self, r):
        ''' read the file header '''
        rc = _ReadContext(r)
        bom = r.readint64()
        if bom != 0x0d0e0a0402080500:
            print("Byteordermark mismatch: ", bom)
            raise RuntimeError()
        rc.itemareastart = r.readint64()
        rc.itemareaend = r.readint64()
        rc.sectioncount = r.readint64()
        n = rc.sectioncount
        while n > 0:
            self.readsection(rc)
            n = n - 1
        bytestoskip = rc.itemareastart - r.position()   # padding bytes between header and item area
        r.skipbytes(bytestoskip)
        return rc

    def readsection(self, rc):
        ''' read a section '''
        r = rc.reader
        sectionid = r.readint32()
        nextsectionoffset = r.readint32()
        beforesection = r.position()
        f = self.getformatter(sectionid)
        f.read(rc)
        aftersection = r.position()
        if (aftersection - beforesection) > nextsectionoffset:
            print("section reads too many bytes")
            raise RuntimeError()

    def writeheader(self, fw, description):
        ''' write the file header '''
        wc = _WriteContext(fw)
        wc.itemareastart = 32
        wc.itemareaend = 0     # no preallocation
        wc.description = description
        wc.sectioncount = 0
        sectionbytes = self.createsections(wc)

        fw.writeint64(0x0d0e0a0402080500)
        fw.writeint64(wc.itemareastart)
        fw.writeint64(wc.itemareaend)
        fw.writeint64(wc.sectioncount)
        wc.writer.writeraw(sectionbytes)

        return wc

    def createsections(self, wc):
        ''' writing the sections into the file raises a small challange:
        before writing the first section, we need to know how many sections will follow, as the
        file format prefixes the section count before the actual sections.
        This implementation accomplishes this by writing the header first into memory, afterwards
        the sectioncount and the sections. Alternatives would be to move the file pointer or
        to enhance the sectionformatters such that they provide the section length without
        writing the section.
        '''
        saved = wc.writer
        sectionstream = BytesIO()
        sectionwriter = _FormattedWriter(_FileIO(sectionstream))
        pos = 32   # sections start at byte position 32
        for formatter in self.sectionformatters:
            payloadstream = BytesIO()
            wc.writer = _FormattedWriter(_FileIO(payloadstream))
            formatter.write(wc)
            payload = payloadstream.getvalue()
            size = len(payload)
            if size > 0:
                # section id
                sectionwriter.writeint32(formatter.id)
                pos += 4

                # nextSectionOffset
                sectionwriter.writeint32(size)
                pos += 4

                # payload
                sectionwriter.writeraw(payloadstream.getvalue())
                pos += size    # no padding or spacing done here

                wc.sectioncount += 1

        # padding
        paddingbytes = 8 - pos % 8
        if paddingbytes == 8:
            paddingbytes = 0
        if paddingbytes:
            padding = b"\0" * paddingbytes
            sectionwriter.writeraw(padding)
        wc.itemareastart = pos + paddingbytes  # first item starts padded on 8 byte boundary.

        wc.writer = saved
        return sectionstream.getvalue()

if __name__ == '__main__':
    import doctest, teafiles.teafile
    doctest.testmod(teafiles.teafile, optionflags = doctest.ELLIPSIS)
