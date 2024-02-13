function As = pauli_measurements(n)
%
%Generates a full set of Pauli measurement masks on n qubits
%
%AJT 12/7/19
%
%inputs: n (number of qubits)
%
%outputs: As (3D tensor of size 2^d x 2^d x 4^d whose slices are the 4^d
%Pauli measurement masks

%Pauli spin matrices
s0 = eye(2);
sx = [0 1; 1 0];
sy = [0 -1i; 1i 0];
sz = [1 0; 0 -1];

As = zeros(2^n,2^n,4^n);

%generate each matrix as a Kronecker product, using binary coding
for j = 1:4^n
    bin = dec2bin(j-1,2*n)=='1';
    A = 1;
    for i = 1:n
        a = rand;
        if (bin(2*(i-1)+1)==0 & bin(2*(i-1)+2)==0)
            si = s0;
        elseif (bin(2*(i-1)+1)==0 & bin(2*(i-1)+2)==1)
            si = sx;
        elseif (bin(2*(i-1)+1)==1 & bin(2*(i-1)+2)==0)
            si = sy;
        else
            si = sz;
        end
        A = kron(si,A);
        % if j == 3
        %     % disp(si)
        %     disp(A)
        % end
    end
    % disp(A)
    As(:,:,j) = A;
    % if j == 10
    %     assert(0==1)
    % end
end

