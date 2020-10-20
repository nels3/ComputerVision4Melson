import cv2
import os
import time
import shutil

# Output directories
directory = 'images'
box_file = 'boxes.txt'

frame_width = 1920
frame_height = 1080

cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Image', frame_width, frame_height)
cv2.moveWindow("Image", 0, 0)

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

# Initalize sliding window's variables
window_x = 0
window_y = 0
window_width = 190
window_height = 190

# Numbers of frames that will be skiped so not many duplicates occurs in training dataset
skip_frames = 3

frame_gap = 0
image_counter = 0
	 
# Getting current image number if data exists
if os.path.exists(box_file):
    # If cleanup is false then we must append the new boxes with the old
    with open(box_file,'r') as text_file:
        box_content = text_file.read()
		
    image_counter = int(box_content.split(':')[-2].split(',')[-1]) + 1

if not os.path.exists(directory):
   os.mkdir(directory)
box_file_writer = open(box_file, 'a')

# Initial wait before getting data
initial_wait = 0

# Start the loop for the sliding window
while(True):
    ret, frame = cap.read()
    if not ret:
        break
        
    frame = cv2.flip( frame, 1 )	
    original = frame.copy()   
	
    if initial_wait > 60:
        frame_gap +=1 
        
        # Move the window to the right  
        if window_x + window_width < frame.shape[1]:
            window_x += 4
            time.sleep(0.1)           
		# Move to the next row
        elif window_y + window_height + 240 < frame.shape[1]:
            window_y += 80   
            window_x = 0
			
            frame_gap = 0
            initial_wait = 0
	    # Break when cannot go further right or down
        else:
            break
    # waiting till initial wait is bigger than thresholg
    else:
        initial_wait += 1

     # Save the n-th image
    if frame_gap == skip_frames:
        color = (0, 255, 0)
        img_name = str(image_counter)  + '.png'
        img_fullname = directory + '/' + str(image_counter) +  '.png'
		
        cv2.imwrite(img_fullname, original)        
        box_file_writer.write('{}:({},{},{},{}),'.format(image_counter, window_x, window_y, window_x + window_width, window_y + window_height))
        
        image_counter += 1  
        frame_gap = 0
    else:
        if initial_wait > 60:
            color = (0, 255, 200)
        else:
            color = (255, 0, 0)
            
		
    # Draw the sliding window
    cv2.rectangle(frame, (window_x, window_y), (window_x + window_width, window_y + window_height), color, 2)
	
    # Display the frame
    cv2.imshow('Image', frame)
    if cv2.waitKey(1) == ord('q'):
        break
        	 
# Release camera and close the file and window
cap.release()
cv2.destroyAllWindows()
box_file_writer.close()

