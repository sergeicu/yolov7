import os 
import shutil
import glob 

from identify_red_v2 import extract_red_bounding_box_coordinates

d='/home/ch215616/w/code/llm/experiments/yolov7/yolov7/elbow_fracture/'

folders = ['Test_annotated_elbow_cases_jpg', 'Test_annotated_elbow_June_cases_jpg']

# detect bounding box 

for folder in folders:
    patients = glob.glob(d+ folder + '/*')
    for patient in patients: 
        patientb=os.path.basename(patient)
        files = glob.glob(patient + '/*.jpg')
        for file in files:
            print(file)
            # from IPython import embed; embed()
            # detect bounding box 
            coordinates = extract_red_bounding_box_coordinates(file)
            fileb=os.path.basename(file)
            if coordinates:
                # copy the labelled file 
                savename = d+'/images_labelled/' + patientb + "___" + fileb
                if not os.path.exists(savename):
                    _ = shutil.copyfile(file,savename)
                    
                # copy unlabelled file 
                savename2 = d+'/images_unlabelled/' + patientb + "___" + fileb
                folder2=folder.replace('_annotated' , '')
                file2=d+folder2+'/'+patientb + '/' + fileb
                assert os.path.exists(file2)
            
                if not os.path.exists(savename2):
                    _ = shutil.copyfile(file2,savename2)
                    
            else: 
                # copy folder where the images have no bounding boxes 
                
                # copy the labelled file 
                savename = d+'/images_nofracture_visible/' + patientb + "___" + fileb
                if not os.path.exists(savename):
                    _ = shutil.copyfile(file,savename)                

                
                
                
            
        