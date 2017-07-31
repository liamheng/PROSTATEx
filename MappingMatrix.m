%% This code transforms patient position to image location
function M=MappingMatrix(PatientPosition,PatientOrientation,VoxelSpacing)
%load the header
% info = dicominfo(filename, 'dictionary', yourvendorspecificdictfilehere);

nSl = 1;%double(info.InstanceNumber);
% nY = double(info.Height);
% nX = double(info.Width);
T1 = double(PatientPosition);

%load pixel spacing / scaling / resolution
RowColSpacing = double(VoxelSpacing);
%of inf.PerFrameFunctionalGroupsSequence.Item_1.PrivatePerFrameSq.Item_1.Pixel_Spacing;
dx = double(RowColSpacing(1));
dX = [1; 1; 1].*dx;%cols
dy = double(RowColSpacing(2));
dY = [1; 1; 1].*dy;%rows
dz = double(RowColSpacing(3));%inf.PerFrameFunctionalGroupsSequence.Item_1.PrivatePerFrameSq.Item_1.SliceThickness; %thickness of spacing?
dZ = [1; 1; 1].*dz;

%directional cosines per basis vector
dircosXY = double(PatientOrientation);
dircosX = dircosXY(1:3);
dircosY = dircosXY(4:6);
if nSl == 1;
    dircosZ = cross(dircosX,dircosY);%orthogonal to other two direction cosines!
else
    N = nSl;%double(inf.NumberOfFrames);
    TN = double(-eval(['inf.PerFrameFunctionalGroupsSequence.Item_',sprintf('%d', N),'.PlanePositionSequence.Item_1.ImagePositionPatient']));
    dircosZ = ((T1-TN)./nSl)./dZ;
end

%all dircos together
dimensionmixing = [dircosX dircosY dircosZ];

%all spacing together
dimensionscaling = [dX dY dZ];

%mixing and spacing of dimensions together
R = dimensionmixing.*dimensionscaling;%maps from image basis to patientbasis

%offset and R together
A = [[R T1];[0 0 0 1]];
M = pinv(A);
%you probably want to switch X and Y
%(depending on how you load your dicom into a matlab array)
% Aold = A;
% A(:,1) = Aold(:,2);
% A(:,2) = Aold(:,1);
