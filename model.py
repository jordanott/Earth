import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, LearningRateScheduler

from stored_dictionaries.data_options import data_opts

OPTS = {
    'adam' : Adam,
    'sgd'  : SGD,
    'rmsprop' : RMSprop,
}

class Network:

    def __init__(self, args, ID):
        self.ID = ID
        self.args = args
        self.build_model()

    def build_model(self):
        
        # MODEL INPUT
        x = input = Input(
            shape=(data_opts[self.args['data']]['input_shape'],)
        )
        
        if self.args['noise'] != 0:
            x = GaussianNoise(self.args['noise'])(x)
        
        # NUMBER OF LAYERS
        for i in range(self.args['num_layers']):
            
            # LAYER WITH LEAKY RELU ACTIVATION
            x = Dense(self.args['num_dense_nodes'])(x)
            x = LeakyReLU(alpha=self.args['leaky_relu'])(x)
            
            # REGULARIZATION
            if self.args['dropout'] != 0: x = Dropout(self.args['dropout'])(x)
            if self.args['batch_norm']: x = BatchNormalization()(x)

        # LAST LAYER OUTPUT
        x = Dense(data_opts[self.args['data']]['output_shape'])(x)
        
        # MODEL CONSTRUCTION
        model = Model(inputs=input, outputs=x)
        self.model = self._compile(model)

    def _compile(self, model):

        model.compile(
            loss='mse',
            optimizer=OPTS[self.args['optimizer']](lr=self.args['lr']),
            metrics=['mse']
        )
        return model

    def train(self, train_gen, valid_gen, trial=None, client=None):
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.args['patience']
            ),
            ModelCheckpoint(
                self.get_model_path() + '.h5',
                save_best_only=True,
                save_weights_only=False
            )
        ]

        if trial is not None:
            callbacks.append(
                client.keras_send_metrics(
                    trial,
                    objective_name='val_loss',
                    context_names=['loss', 'val_loss']
                )
            )

        if True: # self.args['alg'] != 'baseline'
            def schedule(epoch, lr):
                return lr * self.args['lr_decay']

            callbacks.append(
                LearningRateScheduler(schedule)
            )
        
        
        history = self.model.fit_generator(
            train_gen,
            steps_per_epoch=train_gen.n_batches,
            epochs=self.args['epochs'],
            verbose=2,
            validation_data=valid_gen,
            validation_steps=valid_gen.n_batches,
            max_queue_size=50, 
            workers=16,
            use_multiprocessing=True, 
            callbacks=callbacks,
        )
#         for epoch in range(self.args['epochs']):
            
#             history = self.model.fit_generator(
#                 train_gen,
#                 steps_per_epoch=train_gen.n_batches,
#                 epochs=1,
#                 verbose=2,
#                 callbacks=callbacks,
#             )
            
#             if (epoch+1) % 2 == 0:
#                 val_result = self.model.evaluate_generator(
#                     steps=valid_gen.n_batches, 
#                     callbacks=None, 
#                     max_queue_size=50, 
#                     workers=12,
#                     use_multiprocessing=True, 
#                 )
                
#                 client.send_metrics(
#                     trial,
#                     epoch,
#                     val_result[0],
#                     context={'loss':float(history['loss'][0])}
#                 )
                    
                

    def get_model_path(self):
        return self.args['model_dir'] + '%05d' % self.ID

    def save(self, file_name=None):
        if file_name is None:
            file_name = self.get_model_path()

        # save to h5 file
        self.model.save(file_name + '.h5')