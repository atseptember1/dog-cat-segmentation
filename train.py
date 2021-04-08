import tensorflow as tf
import numpy as np
import argparse
import os

from data import Dataset
from model import Unet
from tensorflow.keras import optimizers, losses, metrics
from helper import progressBar


class UnetTrainer:

    def __init__(self, model, learning_rate=0.001, checkpoint_dir="model/checkpoints"):

        self.ckpt = tf.train.Checkpoint(epoch=tf.Variable(1),
                                        val_loss=tf.Variable(np.inf),
                                        optimizer=optimizers.Adam(lr=learning_rate),
                                        model=model)
        self.ckpt_manager = tf.train.CheckpointManager(checkpoint=self.ckpt,
                                                       directory=checkpoint_dir,
                                                       max_to_keep=3)

        self.restore_checkpoint()

        self.loss_fn = losses.SparseCategoricalCrossentropy(from_logits=True)
        self.train_metrics = [metrics.Mean(),
                              metrics.SparseCategoricalAccuracy()]
        self.val_metrics = [metrics.Mean(),
                            metrics.SparseCategoricalAccuracy()]

    @tf.function
    def train_step(self, inputs, labels):
        with tf.GradientTape() as tape:
            y_pred = self.ckpt.model(inputs)
            losses = self.loss_fn(labels, y_pred)
        grads = tape.gradient(losses, self.ckpt.model.trainable_variables)
        self.ckpt.optimizer.apply_gradients(zip(grads, self.ckpt.model.trainable_variables))
        self.train_metrics[0].update_state(losses)
        self.train_metrics[1].update_state(labels, y_pred)

    @tf.function
    def val_step(self, inputs, labels):
        y_pred = self.ckpt.model(inputs)
        losses = self.loss_fn(labels, y_pred)
        self.val_metrics[0].update_state(losses)
        self.val_metrics[1].update_state(labels, y_pred)

    def train_loop(self, train_set, test_set, epochs=20, train_steps=100, validation_steps=50):
        for epoch in range(epochs):
            train_progress = 1
            for X_train, y_train in train_set.take(train_steps):
                self.train_step(X_train, y_train)
                loss_train = self.train_metrics[0].result()
                acc_train = self.train_metrics[1].result()
                progressBar(epoch+1,
                            train_progress,
                            train_steps,
                            loss_train.numpy(),
                            acc_train.numpy())
                train_progress += 1

            val_progress = 1
            for X_test, y_test in test_set.take(validation_steps):
                self.val_step(X_test, y_test)
                loss_val = self.val_metrics[0].result()
                acc_val = self.val_metrics[1].result()
                progressBar(epoch+1,
                            val_progress,
                            validation_steps,
                            loss_val.numpy(),
                            acc_val.numpy(),
                            "Testing")
                val_progress += 1
            print(f"Epoch {epoch+1} - Training Loss: {loss_train} - Training Accuracy: {acc_train} - Testing Loss: {loss_val} - Testing Accuracy: {acc_val}")

            if loss_val < self.ckpt.val_loss:
                print(f"Model improved! Saving checkpoint at epoch {epoch+1}")
                self.ckpt_manager.save()
                self.ckpt.val_loss = loss_val

            self.ckpt.epoch.assign_add(1)
            self.train_metrics[0].reset_states()
            self.train_metrics[1].reset_states()
            self.val_metrics[0].reset_states()
            self.val_metrics[1].reset_states()

    def restore_checkpoint(self):
        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print(f"Model restored at epoch {self.ckpt.epoch.numpy()}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training Unet for Dog-Cat-Segmentation")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs for training")
    parser.add_argument("--train_steps", type=int, default=100, help="Number of training steps per epoch")
    parser.add_argument("--val_steps", type=int, default=50, help="Number of validation steps")
    parser.add_argument("--checkpoint_dir", type=str, default="./model/checkpoint", help="path to write checkpoint")
    parser.add_argument("--weight_path", type=str, default="./model/weights/unet.h5", help="path to save weight")

    args = vars(parser.parse_args())

    unet = Unet()

    train_set = Dataset().get_dataset()
    test_set = Dataset(training=False).get_dataset()

    trainer = UnetTrainer(model=unet,
                          learning_rate=args["lr"],
                          checkpoint_dir=args["checkpoint_dir"])
    trainer.train_loop(train_set=train_set,
                       test_set=test_set,
                       epochs=args["epochs"],
                       train_steps=args["train_steps"],
                       validation_steps=args["val_steps"])

    if not(os.path.exists(os.path.split(args["weight_path"])[0])):
        os.makedirs(os.path.split(args["weight_path"])[0])

    trainer.restore_checkpoint()
    trainer.ckpt.model.save_weights(args["weight_path"])
