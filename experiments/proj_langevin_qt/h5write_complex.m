function h5write_complex(wdata, filename, dataset)
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

fileName       = filename;
DATASET        = dataset;
DIM0           = size(wdata.i);

dims = DIM0;

% size(wdata.i)

%% Create a new file using the default properties.
%
file = H5F.create (fileName, 'H5F_ACC_TRUNC',...
    'H5P_DEFAULT', 'H5P_DEFAULT');

%
%Create the required data types
%
IEEE_F64LE_Type = H5T.copy('H5T_IEEE_F64LE');
sz(1)     =H5T.get_size(IEEE_F64LE_Type);
sz(2)     =H5T.get_size(IEEE_F64LE_Type);


%
% Computer the offsets to each field. The first offset is always zero.
%
offset(1)=0;
offset(2) = sz(1);

%
% Create the compound datatype for memory.
%
memtype = H5T.create ('H5T_COMPOUND', sum(sz));
H5T.insert (memtype,'r',offset(1),IEEE_F64LE_Type);
H5T.insert (memtype,'i',offset(2), IEEE_F64LE_Type);
%
% Create the compound datatype for the file.  Because the standard
% types we are using for the file may have different sizes than
% the corresponding native types, we must manually calculate the
% offset of each member.
%
filetype = H5T.create ('H5T_COMPOUND', sum(sz));
H5T.insert (filetype, 'r', offset(1),IEEE_F64LE_Type);
H5T.insert (filetype, 'i', offset(2), IEEE_F64LE_Type);

%
% Create dataspace.  Setting maximum size to [] sets the maximum
% size to be the current size.
%
flipped_dims = fliplr(dims)
space = H5S.create_simple (length(size(wdata.i)),flipped_dims, flipped_dims);

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


% %
% %% Now we begin the read section of this example.  Here we assume
% % the dataset has the same name and rank, but can have any size.
% %

% %
% % Open file and dataset.
% %
% file = H5F.open (fileName, 'H5F_ACC_RDONLY', 'H5P_DEFAULT');
% dset = H5D.open (file, DATASET);

% %
% % Get dataspace and allocate memory for read buffer.
% %
% space = H5D.get_space (dset);
% [~, dims, ~] = H5S.get_simple_extent_dims (space);
% dims = fliplr(dims);

% %
% % Read the data.
% %
% rdata=H5D.read (dset, memtype, 'H5S_ALL', 'H5S_ALL', 'H5P_DEFAULT');

% %
% % Output the data to the screen.
% %
% for i=1: dims(1)
%     fprintf ('%s[%d]:\n', DATASET, i);
%     fprintf ('Serial number   : %d\n', rdata.serial_no(i));
%     fprintf ('Location        : %s\n', rdata.location{i});
%     fprintf ('Temperature (F) : %f\n', rdata.temperature(i));
%     fprintf ('Pressure (inHg) : %f\n\n', rdata.pressure(i));
% end

% %
% % Close and release resources.
% %
% H5D.close (dset);
% H5S.close (space);
% H5T.close (memtype);
% H5F.close (file);

