import utils_add_data as uad
import os


class ImagePHI:
    def __init__(self, in_folder_path, out_folder_path, csv_data_path):
        self.folder_path = in_folder_path
        self.images = []
        self.csv_data_path = csv_data_path
        self.out_folder_path = out_folder_path

    def load_images(self):
        for root, dirs, files in os.walk(self.folder_path):
            for file_name in files:
                if file_name.lower().endswith((".jpg", ".png", ".jpeg", ".dcm")):
                    #print(f"Loading image: {file_name}")
                    image_path = os.path.join(root, file_name)
                    self.images.append(image_path)
                else:
                    # Skip non-image file
                    print(f"Skipping non-image file: {file_name}")

    def process_images(self):
        # loop through images
        for img in self.images:
            # get pixel level data from different file types: 
            image_to_text = uad.read_image_get_pixels(img)
            if image_to_text.shape[1] > 200:
                # Normalize the uint16 data to 0-255 range and convert to uint8      
                phi_values = uad.select_phi_info(csv_data_path = self.csv_data_path)
                new_image = uad.add_text_to_image(image_to_text, texts = phi_values)
                # save the new image
                file_save_name, csv_filename = uad.generate_random_filename()
                file_save_path = uad.save_images(output_folder=self.out_folder_path, new_filename=file_save_name)
                new_image.save(file_save_path)
                uad.save_list_as_column_to_csv(phi_values, self.out_folder_path, csv_filename)
            else:
                pass

# Create an instance of ImageFolder and specify the folder path
img_2_proc = ImagePHI(in_folder_path = "/Users/ask4118/Documents/Data/PHI_test_data/manifest-1617826555824/Pseudo-PHI-DICOM-Data", out_folder_path = "/Users/ask4118/Documents/Data/PHI_test_data/processed", csv_data_path = '/Users/ask4118/Documents/Data/PHI_test_data/csvs/fake_patient_info.csv')
# Load the images from the folder
img_2_proc.load_images()
# Process the images
img_2_proc.process_images()
