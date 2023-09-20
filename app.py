import cv2
from flask import Flask, request, render_template, send_file
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from joblib import Memory
from PIL import Image
import os

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'

# Load segmentation model
sam = sam_model_registry["vit_h"](checkpoint=r"C:\Users\User\Documents\Academic\Graduate\Research\Labelling tool\Labeling_Tool\Labeling_Tool\segment-anything-main\sam_vit_h_4b8939.pth").to(device="cuda")

# Set up joblib caching
memory = Memory(cachedir="./cache", verbose=0)

# Define function to create mask generator
@memory.cache
def create_mask_generator(stability_score_thresh):
    return SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.9,
        stability_score_thresh=stability_score_thresh,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,
    )

# Define function to display mask
def show_anns(image, masks):
    if len(masks) == 0:
        return
    sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    for i, mask in enumerate(sorted_masks):
        m = mask['segmentation']
        color_mask = np.ones_like(image) * 255
        for j in range(3):
            image[:, :, j] = np.where(m, image[:, :, j] * 0.5 + 0.5 * color_mask[:, :, j], image[:, :, j])
        ax.imshow(image)
    plt.axis('off')
    return fig

# Define route for uploading images
@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            # Save the uploaded image
            image_path = os.path.join('uploads', image_file.filename)
            image_file.save(image_path)
            
            stability_score_thresh = float(request.form.get('stability_score_thresh', 0.96))

            # Create or retrieve the mask generator
            mask_generator = create_mask_generator(stability_score_thresh)

            # Process the image and generate masks
            image = np.array(Image.open(image_path))
            masks = mask_generator.generate(image)
            fig = show_anns(image, masks)

            # Save the processed image with masks
            output_path = os.path.join('uploads', 'output.png')
            fig.savefig(output_path)

            # Return the processed image with masks
            return send_file(output_path, mimetype='image/png')

    return render_template('index.html')

# Define route for editing masks
@app.route('/edit', methods=['POST'])
def edit_mask():
    image_path = os.path.join('uploads', 'output.png')
    image = cv2.imread(image_path)

    # Get the mask editing parameters from the request
    x = int(request.form['x'])
    y = int(request.form['y'])
    radius = int(request.form['radius'])
    erase = bool(int(request.form['erase']))

    # Draw or erase the mask based on the parameters
    if erase:
        cv2.circle(image, (x, y), radius, (0, 0, 0), -1)  # Draw a filled circle with black color to erase the mask
    else:
        cv2.circle(image, (x, y), radius, (255, 255, 255), -1)  # Draw a filled circle with white color to add to the mask

    # Save the edited image
    edited_image_path = os.path.join('uploads', 'edited_output.png')
    cv2.imwrite(edited_image_path, image)

    # Return the edited image
    return send_file(edited_image_path, mimetype='image/png')

if __name__ == '__main__':
    # Define Streamlit app
    app.secret_key = 'supersecretkey'
    app.run(debug=True)

