import cv2
import os
import numpy as np
from cv2 import imread,imwrite
from flask import Flask,render_template,request #,flash

app=Flask(__name__)
app.secret_key="123"

app.config['UPLOAD_FOLDER']="static\\images"

@app.route('/',methods=['GET','POST'])
def upload():
    if request.method=='POST':
        upload_image=request.files['choose_file']

        if upload_image.filename!='':
            fname=upload_image.filename
            filepath1=os.path.join(app.config["UPLOAD_FOLDER"],fname)
            upload_image.save(filepath1)
            repath=''
            #algorithm = request.form['flexRadioDefault']
            repath=denoise(filepath1,fname)
            return render_template('Speckle Noise.html',path1=filepath1,path2=repath)
            #flash("Image Upload Successfully","success")
    return render_template("Speckle Noise.html",path2='')
def denoise(pathio,fname):
    img=cv2.imread(pathio)
    gn_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    lambda_ = 0.1  # Regularization parameter
    niters = 30    # Number of iterations
    brightness=10
    contrast=1
    # Perform TV-L1 denoising
    denoised1 = gn_img.copy()
    #denoised2=solve(denoised1)
    denoised3=cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15);
    denoised3=cv2.addWeighted(denoised3, contrast, np.zeros(denoised3.shape, denoised3.dtype), 0, brightness)
    #denoised3=gaussian_smoothing(denoised1)
    repath='static/images/'
    opfile=fname +'-denoised-img.jpg'
    cv2.imwrite(os.path.join(repath,opfile),denoised3)
    pathres=repath+opfile
    return pathres

def gaussian_smoothing(image):
    # Loading image in form of a matrix
    #image = input_image

    # gaussian mask initiated
    gaussian_mask = np.array(
        [
            [ 1, 1, 2, 2, 2, 1, 1 ],
            [ 1, 2, 2, 4, 2, 2, 1 ],
            [ 2, 2, 4, 8, 4, 2, 2 ],
            [ 2, 4, 8, 16, 8, 4, 2 ],
            [ 2, 2, 4, 8, 4, 2, 2 ],
            [ 1, 2, 2, 4, 2, 2, 1 ],
            [ 1, 1, 2, 2, 2, 1, 1 ]
        ])

    image_height = image.shape[ 0 ]
    image_width = image.shape[ 1 ]

    # Initializing 2-D array to store convolution values
    convoluted_gaussian_arr = np.empty((image_height, image_width))

    for row in range(image_height):
        for col in range(image_width):

            # setting the undefined pixel value to 0
            if 0 <= row < 3 or image_height - 3 <= row <= image_height - 1 or 0 <= col < 3 or image_width - 3 <= col <= image_width - 1:
                convoluted_gaussian_arr[ row ][ col ] = 0
            else:
                x = 0

                # calculating convoluted values for each pixel position and storing it in 'convoluted_gaussian_arr'
                for k in range(7):
                    for l in range(7):
                        x = x + (image[ row - 3 + k ][ col - 3 + l ] * gaussian_mask[ k ][ l ])

                # normalizing the value
                convoluted_gaussian_arr[ row ][ col ] = x / 140

     
    return convoluted_gaussian_arr


def findAllNeighbors(padImg,small_window,big_window,h,w):
    # Finding width of the neighbor window and padded image from the center pixel
    smallWidth = small_window//2
    bigWidth = big_window//2

    # Initializing the result
    neighbors = np.zeros((padImg.shape[0],padImg.shape[1],small_window,small_window))

    # Finding the neighbors of each pixel in the original image using the padded image
    for i in range(bigWidth,bigWidth + h):
        for j in range(bigWidth,bigWidth + w):
            neighbors[i,j] = padImg[(i - smallWidth):(i + smallWidth + 1) , (j - smallWidth):(j + smallWidth + 1)]

    return neighbors

# Function to calculate the weighted average value (Ip) for each pixel
def evaluateNorm(pixelWindow, neighborWindow, Nw):
    # Initialize numerator and denominator of Ip (Ip = Ip_Numerator/Z)
    Ip_Numerator,Z = 0,0

    # Calculating Ip for pixel p using neighborood pixels q
    for i in range(neighborWindow.shape[0]):
      for j in range(neighborWindow.shape[1]):
        # (small_window x small_window) array for pixel q
        q_window = neighborWindow[i,j]

        # Coordinates of pixel q
        q_x,q_y = q_window.shape[0]//2,q_window.shape[1]//2

        # Iq value
        Iq = q_window[q_x, q_y]

        # Norm of Ip - Iq
        w = np.exp(-1*((np.sum((pixelWindow - q_window)**2))/Nw))

        # Calculating Ip
        Ip_Numerator = Ip_Numerator + (w*Iq)
        Z = Z + w

    return Ip_Numerator/Z

def solve(img,h=30,small_window=7,big_window=21):
    # Padding the original image with reflect mode

    img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    padImg = np.pad(img,big_window//2,mode='reflect')
    Nw = (h**2)*(small_window**2)
    h,w = img.shape
    result = np.zeros(img.shape)
    bigWidth = big_window//2
    smallWidth = small_window//2
    neighbors = findAllNeighbors(padImg, small_window, big_window, h, w)
    for i in range(bigWidth, bigWidth + h):
        for j in range(bigWidth, bigWidth + w):
            pixelWindow = neighbors[i,j]
            neighborWindow = neighbors[(i - bigWidth):(i + bigWidth + 1) , (j - bigWidth):(j + bigWidth + 1)]
            Ip = evaluateNorm(pixelWindow, neighborWindow, Nw)
            result[i - bigWidth, j - bigWidth] = max(min(255, Ip), 0)

    return result



@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/signup')
def signup():
    return render_template('Signup.html')
if __name__ == '__main__':
    app.run(debug=False,port=5000)

