clear
close all

[num, Txt]= xlsread('ProstateX-Findings-Train.xlsx', 'A2:L3780');
Name=Txt(:,1);
Finding=num(:,1);
LesionPatientLocation=str2num(char(Txt(:,3)));
%Modulity=[{'BVAL'}];
Modulity=[{'ADC'},{'Tra'}];%,{'Ktrans'},{'Sag'}
for nn=183:183%numel(Name)
    for mm=2:2
        pwd=['Modulity\DICOM\',Modulity{mm},'\'];
		%pwd=['SPIE\DOI\',Name{nn}];
% 		Path=dir(pwd);
%         subPath=dir([pwd,'\',Path(3).name,'\*',Modulity{mm},'*']);
%         SliceID=dir([pwd,'\',Path(3).name,'\',subPath(1).name,'\*.dcm']);%For real data
        SliceID=dir([pwd,Name{nn},'*']);%For fake data
        Dicomdata=[];
        InstanceNumber=[];
        PatientPosition=[];
        PatientOrientation=[];
        for ii=1:numel(SliceID)
%             info = dicominfo([pwd,'\',Path(3).name,'\',subPath(1).name,'\',SliceID(ii).name]);
            info = dicominfo([pwd,SliceID(ii).name]);
            Dicomdata(:,:,ii)=dicomread(info);
            InstanceNumber(ii)=info.InstanceNumber;
            PatientPosition{ii}=info.ImagePositionPatient;
            PatientOrientation{ii}=info.ImageOrientationPatient;
            SliceLocation{ii}=info.SliceLocation;
        end
        [Slice,Order]=sort(InstanceNumber);
        DicomSlice=Dicomdata(:,:,Order);
        PatientPosition=PatientPosition(Order);
        PatientOrientation=PatientOrientation(Order);
        SliceLocation=SliceLocation(Order);
        VoxelSpacing=[info.PixelSpacing;info.SpacingBetweenSlices];
        
        Transform=MappingMatrix(PatientPosition{1},PatientOrientation{1},VoxelSpacing);
        LesionCenter=LesionPatientLocation(nn,:)';
        TransformOutput=Transform*[LesionCenter; 1];
        ImageLocation=TransformOutput;
        %%For ADC Tra
        
        ImagePosition=round(abs([ImageLocation(1);ImageLocation(2)-(size(DicomSlice,1)-size(DicomSlice,2))/4;...
            ImageLocation(3)]))+1;%For ununified size;Matlab do not need direction change		
			
        eval(['Location.',Modulity{mm},'=ImagePosition;'])
        
        %% Mask
        DicomMask=zeros(size(DicomSlice));
        [X,Y,Z] = meshgrid(1:size(DicomSlice,2), 1:size(DicomSlice,1),1:size(DicomSlice,3));
        Distance=((X-ImagePosition(1))*VoxelSpacing(1)).^2+((Y-ImagePosition(2))*VoxelSpacing(1)).^2+((Z-ImagePosition(3))*VoxelSpacing(3)).^2;
        for Map=30:-1:1
            DicomMask(find(Distance<=Map^2))=Map;
        end
        
        eval([Modulity{mm},'=DicomMask;'])
        nii_img = make_nii(flip(flip(permute(DicomSlice, [2 1 3]),2),3),VoxelSpacing');
        save_nii(nii_img, ['NIFTITestingData/',Name{nn},'_',Modulity{mm},'.nii.gz']);
%         nii_img = make_nii(flip(flip(permute(DicomMask, [2 1 3]),2),3),VoxelSpacing');
%         save_nii(nii_img, ['NIFTITrainingData/',Name{nn},'_',Modulity{mm},'_Mask_',num2str(Finding(nn)),'.nii.gz']);        
        
    end
%save(['MaskData\',Name{nn},'_Mask_',num2str(Finding(nn)),'.mat'],'ADC','Sag','Tra','Ktrans','Location') 

end


