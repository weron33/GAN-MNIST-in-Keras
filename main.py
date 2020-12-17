from ml_model import AdversarialModel
from data_operations import BatchedData
import numpy as np

# Prepare data (downloading, batching and choosing)
batch_size = 16
batched_data = BatchedData(batch_size=batch_size)
batched_data.next_batch()

# Creating adversarial model that contains Generator and Discriminator
adversarial_model = AdversarialModel(epochs=500)
adversarial_model.compile()

# Generate noise to check out how fake images looks for now
noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
images_fake = adversarial_model.generate_images(noise)

# Model training
adversarial_model.train(100)

# Generate noise once more to check out how the results changed
noise = np.random.uniform(-1.0, 1.0, size=[16, 100])
image = adversarial_model.generate_images(noise)
