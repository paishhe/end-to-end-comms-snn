## Broad Code Structure

**tx.py** : has the transmitter side encoder block
**synthetic_bit_data.py** : generates a synthetic dataset for system training
**channel.py** : implements a simple awgn channel
**c_extractor.py** : implements the channel extractor block, it is the original ann implementation
**c_extractor_cnn.py** : implements the channel extractor block, it is my snn implementation
**rx.py** : has the receiver side decoder block

**main.py** : the main script to run the simulation

