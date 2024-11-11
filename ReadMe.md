

To create the data run:

python create_data.py

this will make plots in 'data_images' and save the data as .npy files in 'data'

The .npy files are loaded in the scripts corresponding to the models, e.g. VAE, GP, PCA
(heteroscdastic VAE, GpLVM  and VAE_varying_noise do not yet work as they should)




Run 

python VAE.py 

to train a VAE on the data. 
After the specified number of epochs, an image containing the data and generations is saved in 'images'