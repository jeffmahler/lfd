function F = makeLWfilters
% creates the Laws texture masks

L5=[1,4,6,4,1];
E5=[-1,-2,0,2,1];
S5=[-1,0,2,0,-1];
R5=[1,-4,6,-4,1];

filt_size = 5;
num_filts = 15;
F = zeros(filt_size, filt_size, num_filts);

F(:,:,1)=L5'*E5;
F(:,:,2)=E5'*L5;
F(:,:,3)=L5'*R5;
F(:,:,4)=R5'*L5;
F(:,:,5)=E5'*S5;
F(:,:,6)=S5'*E5;
F(:,:,7)=S5'*S5;
F(:,:,8)=R5'*R5;
F(:,:,9)=L5'*S5;
F(:,:,10)=S5'*L5;
F(:,:,11)=E5'*E5;
F(:,:,12)=E5'*R5;
F(:,:,13)=R5'*E5;
F(:,:,14)=S5'*R5;
F(:,:,15)=R5'*S5; 

for i = 1:15
   F(:,:,i) = normalize(F(:,:,i)); 
end
  
end

function f=normalize(f), f=f-mean(f(:)); f=f/sum(abs(f(:))); return; end



