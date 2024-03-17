function h5ex_t_cmpd
%**************************************************************************
%
%  This example shows how to read and write compound
%  datatypes to a dataset.  The program first writes
%  compound structures to a dataset with a dataspace of DIM0,
%  then closes the file.  Next, it reopens the file, reads
%  back the data, and outputs it to the screen.
%
%  This file is intended for use with HDF5 Library version 1.8
%**************************************************************************

fileName       = 'h5ex_t_cmpd.h5';
DATASET        = 'DS1';
DIM0           = 4;

dims = DIM0;

%
% Initialize data. It is more efficient to use Structures with array fields
% than arrays of structures.
%
wdata.serial_no   =int32([1153 ; 1184 ; 1027  ;    1313]);
wdata.location    ={'Exterior (static)', 'Intake',...
    'Intake manifold', 'Exhaust manifold'};
wdata.temperature =[53.23; 55.12; 130.55; 1252.89];
wdata.pressure    =[24.57; 22.95;  31.23;   84.11];
%
%% Create a new file using the default properties.
%
file = H5F.create (fileName, 'H5F_ACC_TRUNC',...
    'H5P_DEFAULT', 'H5P_DEFAULT');

%
%Create the required data types
%
intType   =H5T.copy('H5T_NATIVE_INT');
sz(1)     =H5T.get_size(intType);
strType   = H5T.copy ('H5T_C_S1');
H5T.set_size (strType, 'H5T_VARIABLE');
sz(2)     =H5T.get_size(strType);
doubleType=H5T.copy('H5T_NATIVE_DOUBLE');
sz(3)     =H5T.get_size(doubleType);
doubleType=H5T.copy('H5T_NATIVE_DOUBLE');
sz(4)     =H5T.get_size(doubleType);


%
% Computer the offsets to each field. The first offset is always zero.
%
offset(1)=0;
offset(2:4)=cumsum(sz(1:3));

%
% Create the compound datatype for memory.
%
memtype = H5T.create ('H5T_COMPOUND', sum(sz));
H5T.insert (memtype,...
    'serial_no',offset(1),intType);
H5T.insert (memtype,...
    'location',offset(2), strType);
H5T.insert (memtype,...
    'temperature',offset(3), doubleType);
H5T.insert (memtype,...
    'pressure',offset(4), doubleType);

%
% Create the compound datatype for the file.  Because the standard
% types we are using for the file may have different sizes than
% the corresponding native types, we must manually calculate the
% offset of each member.
%
filetype = H5T.create ('H5T_COMPOUND', sum(sz));
H5T.insert (filetype, 'serial_no', offset(1),intType);
H5T.insert (filetype, 'location', offset(2), strType);
H5T.insert (filetype, 'temperature',offset(3), doubleType);
H5T.insert (filetype, 'pressure',offset(4), doubleType);


%
% Create dataspace.  Setting maximum size to [] sets the maximum
% size to be the current size.
%
space = H5S.create_simple (1,fliplr( dims), []);

%
% Create the dataset and write the compound data to it.
%
dset = H5D.create (file, DATASET, filetype, space, 'H5P_DEFAULT');
H5D.write (dset, memtype, 'H5S_ALL', 'H5S_ALL', 'H5P_DEFAULT', wdata);

%
% Close and release resources.
%
H5D.close (dset);
H5S.close (space);
H5T.close (filetype);
H5F.close (file);


%
%% Now we begin the read section of this example.  Here we assume
% the dataset has the same name and rank, but can have any size.
%

%
% Open file and dataset.
%
file = H5F.open (fileName, 'H5F_ACC_RDONLY', 'H5P_DEFAULT');
dset = H5D.open (file, DATASET);

%
% Get dataspace and allocate memory for read buffer.
%
space = H5D.get_space (dset);
[~, dims, ~] = H5S.get_simple_extent_dims (space);
dims = fliplr(dims);

%
% Read the data.
%
rdata=H5D.read (dset, memtype, 'H5S_ALL', 'H5S_ALL', 'H5P_DEFAULT');

%
% Output the data to the screen.
%
for i=1: dims(1)
    fprintf ('%s[%d]:\n', DATASET, i);
    fprintf ('Serial number   : %d\n', rdata.serial_no(i));
    fprintf ('Location        : %s\n', rdata.location{i});
    fprintf ('Temperature (F) : %f\n', rdata.temperature(i));
    fprintf ('Pressure (inHg) : %f\n\n', rdata.pressure(i));
end

%
% Close and release resources.
%
H5D.close (dset);
H5S.close (space);
H5T.close (memtype);
H5F.close (file);

