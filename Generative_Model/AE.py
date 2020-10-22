import torch 
import torch.nn as nn 
from torch.autograd import Variable 
from MOFVAE import Auto_Encoder as AE 
import util
from mof_dataset import MOFDataset
import numpy as np
from pathlib import Path


cuda = True if torch.cuda.is_available() else False 
device = torch.device('cuda' if cuda else 'cpu')
num_epochs = 1000
batch_size = 128
learning_rate = 1e-3

dataloader = MOFDataset.get_data_loader("../3D_Grid_Data/Training_MOFS.p",  batch_size, no_grid=True)

model = None
saved_model = Path("AE_MODEL.p")
if (saved_model.is_file()):
	print("Loading Saved Model")
	model= torch.load("AE_MODEL.p")
else:
	model = AE.ConvolutionalAE(2048,11)

if cuda:
	model.cuda()


criterion = nn.MSELoss(reduction = 'sum')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
fake_data = Variable(torch.ones(1,11,32,32,32)).cuda()
 

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 11, 32, 32,32)
    return x

def train():
	loss = 0
	total_loss = 0
	for data in enumerate(dataloader):
		batch, img= data
		# print(img.shape)
		# img = img.view(img.size(0), -1)
		# print(img.shape)
		img = Variable(img.float()).cuda()
		# # ===================forward=====================
		output = model(img)	        

		loss = criterion(output.flatten(), img.flatten())

		print("Batch [{}/{}], loss:{:.6f}".format(batch, len(dataloader), loss.item()))
		total_loss += loss.item()
		# # ===================backward====================
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
    # ===================log========================
	total_loss = total_loss / (len(dataloader))
	return total_loss     

def small_batch_train(batch_of_data):
	loss = 0 
	img = Variable(batch_of_data.float()).cuda()
	output = model(img)
	loss = criterion(output, img)
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

	return loss.item()

def visualize(tensor, savefile):
	channels = list(range(11))
	util.Visualize_MOF(tensor, channels, savefile=savefile)

def trial(data, file_name):
	with torch.no_grad():
		output = model(data.float().cuda())
		# loss = criterion(fake_data, output)

		print("Outputing file: ", file_name)
		visualize(output[0].cpu().numpy(), file_name)
	# optimizer.zero_grad()
	# loss.backward()
	# optimizer.step()

	# return loss 

def main():

	min_loss =0 
	test_image = None
	img = None
	for data in enumerate(dataloader):
		batch, img = data 
		print(img[0].shape)
		img[img < 10E-2] = 0
		test_image = img.clone().detach()
		visualize(test_image[0], "MOF.png")
		break

	# trial()
	
	for epoch in range(num_epochs):
		loss = train()
		# print(epoch, loss.item())


		if (epoch %1 ==0):
			print('epoch [{}/{}], loss:{:.6f}'.format(epoch + 1, num_epochs, loss))
		if(epoch % 10 == 0):
			trial(test_image, "Decoded_Image_Prog_"+str(epoch)+".png")

		if (epoch % 50 ==0):
			torch.save(model, "AE_MODEL_FULL.p")
	# with torch.no_grad():
		# ones = Variable(torch.ones(1,12,32,32,32)).cuda()
		# output = model(ones)

		# loss1 = criterion(output, ones)

		# loss2 = criterion(output.flatten(),ones.flatten() )

		# print(loss1.item(), loss2.item()) 
		# print(output[0].shape)
		# print(output[0][0])


	
main()