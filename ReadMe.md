

To create the data run:

python create_data.py

this will make plots in 'data_images' and save the data as .npy files in 'data'

The .npy files are loaded in the scripts corresponding to the models, e.g. VAE, GP, PCA
(heteroscedastic VAE, GpLVM  and VAE_varying_noise do not yet work as they should)




Run 

python VAE.py 

to train a VAE on the data. 
After the specified number of epochs, an image containing the data and generations is saved in 'images'


My experiments were mainly focused on the simple VAE and comparisons to PCA.
I find that the VAE can not handle distributions with varying noise levels. Neither can a GP. Both are nonlinear models for
systems of the form 
x = f(z) + e
and struggle with
x = f(z) + e(z)
.

Furthermore, for concentric circles and spiral distribution, the pattern can be learned only up to some degree.
Maybe architectural changes (using a transformer etc.) could improve this.
Most likely a better prior can do the job better. 
Compare the concetric circles to the single circle generations (which are quite good)
and note that the kernel trick enables construction of a seperating hyplerplane.
Thus learning two seperated priors could be possible. (For example, by classification and then training two VAEs on the classified data)