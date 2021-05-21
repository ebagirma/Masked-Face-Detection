# Live UnMasked Face Recognition 
## Introduction

This is a live face unmasked face recogination from webcam (or any camera for that matter). It will detect unmasked face and tries to indentify the indetity of that person.  



## Dependencies

**Packages needed are found in the requirements:**

    $ pip3 install -r requrements.txt



## Task 1: Un Masked face detection

This is indetifying faces with no mask. Which is more or less done. `more Detiles will be written. `

    $ python3 detect_mask_video.py





## Task 2: Recognition of face

After detecting which face is not waring a mask, the next task is to indetify who it is. For this task, we collected our custom data's and trained our own face recognation based on transfer learning off of `MobileNetV2, imagenet dataset` on the data we collected. Like wise you can add your own datasets/faces to the data set by using:

    $ python3 collect-dataset-for-face-recognation.py <your-name>

After adding your own datasets. you can train the model.

    $ python3 face_Recognition_training.py



## Task 3

After we are done with this, we will incorporate it with React.Js framework for having a good visiual desiply. 



## Note: 
Forgive all the misspellings you might see. Its work in progress

