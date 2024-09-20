from cleanvision import Imagelab

# Path to your dataset, you can specify your own dataset path
dataset_path = "/home/areebadnan/Areeb_code/work/Visua_Data/output_videos/65ae82fb939a1d114f107f72/images"

# Initialize imagelab with your dataset
imagelab = Imagelab(data_path=dataset_path)

# Find issues
imagelab.find_issues()

imagelab.report()