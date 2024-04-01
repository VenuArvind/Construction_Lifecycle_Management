from model import *
import pickle 
import cv2 
from helpers import * 

model_name= "resnet34"
num_classes = 1024 

weight_path = "C:\\Users\\kumaran\\Desktop\\fyp_dataset\\data\\resnet34_stanford.pth"

encoder = initialize_encoder(model_name, num_classes,use_pretrained=True)
# Full model
model_ft = SegNet(encoder, num_classes)
model_dict = torch.load(weight_path, map_location=torch.device('cpu'))

model_ft.load_state_dict(model_dict)

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Send the model to GPU
model_ft = model_ft.to(device)

def prediction(pkl_path):
    pkl = pickle.load(open(pkl_path, 'rb'))
    img = pkl['image'].astype('float32')
    label = pkl['edge'].astype('float32')
    label2 = pkl['junc'].astype('float32')
    mask = pkl['line'].astype('float32')
    filename = pkl['filename']

    #cv2.imshow("",mask)

    cv2.waitKey(0)

    # lr flip
    img2 = np.fliplr(img).copy()
    mask2 = np.fliplr(mask).copy()

    image = torch.tensor(img).to(device).float()
    labels = torch.tensor(label).to(device).float()
    labels2 = torch.tensor(label2).to(device).float()
    masks = torch.tensor(mask).to(device).float()     
    
    inputs = image.permute(2,0,1)
    inputs = inputs.unsqueeze(0)
    masks = masks.permute(2,0,1)
    masks = masks.unsqueeze(0)
    inputs = torch.cat((inputs,masks),1)
    labels = labels.permute(2,0,1)
    labels = labels.unsqueeze(0)
    labels2 = labels2.permute(2,0,1)
    labels2 = labels2.unsqueeze(0)

    image2 = torch.tensor(img2).to(device).float()
    masks2 = torch.tensor(mask2).to(device).float()

    inputs2 = image2.permute(2,0,1)
    inputs2 = inputs2.unsqueeze(0)
    masks2 = masks2.permute(2,0,1)
    masks2 = masks2.unsqueeze(0)
    inputs2 = torch.cat((inputs2,masks2),1)

    inputs = torch.cat((inputs, inputs2),0)

    # forward
    outputs, outputs2 = model_ft(inputs)

    outputs1 = outputs[1]
    outputs22 = outputs2[1]


    inv_idx = torch.arange(outputs1.size(2)-1, -1, -1).to(device).long()
    outputs1 = outputs1.index_select(2, inv_idx)
    outputs = torch.mean(torch.cat((outputs[0].unsqueeze(0), outputs1.unsqueeze(0)), 0), 0, True)

    outputs22 = outputs22.index_select(2, inv_idx)
    outputs2 = torch.mean(torch.cat((outputs2[0].unsqueeze(0), outputs22.unsqueeze(0)), 0), 0, True)

    cor_img = outputs2.data.cpu().numpy()
    edg_img = outputs.data.cpu().numpy()

    #cv2.imshow("",cor_img[0][0])
    #cv2.waitKey(0)

    input_tensor = cor_img[0]  # Random values for demonstration

# Permute the dimensions to (512, 1024, 3) for RGB image representation
    rgb_image = np.transpose(input_tensor, (1, 2, 0))

    #cv2.imshow("",rgb_image)

    #cv2.waitKey(0)

    rgb_image2 = rgb_image*0.5 + img*0.5 

    original_cor = pkl["cor"]

    predicted_cor = get_initial_corners(cor_img, 21, 3) 

    for i in original_cor:
        cv2.circle( rgb_image2 , np.array(i).astype("int32") , 10 , (255,0,0), -1 ) 

    for i in predicted_cor:
        cv2.circle( rgb_image2, i , 10 , (255,255,0), -1) 

    cv2.imshow("image",rgb_image2)
    cv2.waitKey(0)
    # cv2.imshow("",rgb_image)


# # Define the dimensions of the output 3D representation
# output_width = 1024
# output_height = 768  # Adjust this value as needed

# # Define the corresponding points in the output 3D representation
# output_points = np.array([[0, 0], [output_width, 0], [output_width, output_height], [0, output_height]], dtype=np.float32)

# # Calculate the perspective transformation matrix
# perspective_matrix = cv2.getPerspectiveTransform(corner_points, output_points)

# # Read the panorama image
# panorama_image = cv2.imread('panorama_image.jpg')

# # Apply the perspective transformation to the panorama
# transformed_panorama = cv2.warpPerspective(panorama_image, perspective_matrix, (output_width, output_height))

# # Display the transformed panorama
# cv2.imshow('Transformed Panorama', transformed_panorama)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#     cv2.waitKey(0)

prediction("C:\\Users\\kumaran\\Desktop\\fyp_dataset\\data\\train_stanford\\area1_0024.pkl")

# first_layer = next(iter(model_ft.parameters()))

# # Get the shape of the weight tensor of the first layer
# input_shape = first_layer.shape

# print("Input shape:", input_shape)


# # Inspect the keys of the state dictionary to identify the last layer
# last_layer_name = None
# for key in model_dict.keys():
#     if key.endswith('weight'):  # Assuming the weight tensor indicates a layer
#         last_layer_name = key
#         break

# if last_layer_name:
#     last_layer_size = model_dict[last_layer_name].shape  # Assuming the first dimension indicates the output size
#     print("Output layer size:", last_layer_size)
# else:
#     print("Unable to determine the last layer of the model.")