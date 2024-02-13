function ret = h5(mat, name)
    filename = strcat(name, "_m.h5");
    coder.rowMajor
    % S = struct('r', real(mat), 'i', imag(mat));
    if isfile(filename)
        delete (filename)
    end
    % S
    real_mat = real(mat);
    imag_mat = imag(mat);
    real_mat((real_mat == 0)) = 0;
    imag_mat((imag_mat == 0)) = 0;
    h5create(filename, "/data_real", size(mat))
    h5create(filename, "/data_imag", size(mat))
    h5write(filename, "/data_real", real_mat)
    h5write(filename, "/data_imag", imag_mat)

    % h5write_complex(S, filename, "/data")
    % h5create(filename,"/data", size(S))
    % % h5write(filename, "/data", S)
    % struct2hdf5(S, "/data", "", filename)
end