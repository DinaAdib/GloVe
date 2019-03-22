from libraries import *
import torch
import torch.optim as optim

embedSize = 2

# Weight function
def wf(x):
	if x < xmax:
		return (x/xmax)**alpha
	return 1

# Set up word vectors and biases
l_embed, r_embed = [
	[Variable(torch.from_numpy(np.random.normal(0, 0.01, (embedSize, 1))),
		requires_grad = True) for j in range(vocabSize)] for i in range(2)]
l_biases, r_biases = [
	[Variable(torch.from_numpy(np.random.normal(0, 0.01, 1)),
		requires_grad = True) for j in range(vocabSize)] for i in range(2)]

# Set up optimizer
optimizer = optim.Adam(l_embed + r_embed + l_biases + r_biases, lr = learningRate)

# Batch sampling function
def gen_batch():
	sample = np.random.choice(np.arange(len(coocs)), size=batch_size, replace=False)
	l_vecs, r_vecs, covals, l_v_bias, r_v_bias = [], [], [], [], []
	for chosen in sample:
		ind = tuple(cooccurrences[chosen])
		l_vecs.append(l_embed[ind[0]])
		r_vecs.append(r_embed[ind[1]])
		covals.append(cooccurrences[ind])
		l_v_bias.append(l_biases[ind[0]])
		r_v_bias.append(r_biases[ind[1]])
	return l_vecs, r_vecs, covals, l_v_bias, r_v_bias

# Train model
for epoch in range(numEpochs):
	num_batches = int(len(cooccurrences)/batchSize)
	avg_loss = 0.0
	for batch in range(num_batches):
		optimizer.zero_grad()
		l_vecs, r_vecs, covals, l_v_bias, r_v_bias = gen_batch()
		# For pytorch v2 use, .view(-1) in torch.dot here. Otherwise, no need to use .view(-1).
		loss = sum([torch.mul((torch.dot(l_vecs[i].view(-1), r_vecs[i].view(-1)) +
				l_v_bias[i] + r_v_bias[i] - np.log(covals[i]))**2,
				wf(covals[i])) for i in range(batchSize)])
		avg_loss += loss.data[0]/num_batches
		loss.backward()
		optimizer.step()
	print("Average loss for epoch "+str(epoch+1)+": ", avg_loss)
