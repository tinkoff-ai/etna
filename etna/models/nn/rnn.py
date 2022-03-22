import torch
import torch.nn as nn
from pytorch_lightning import LightningModule

from etna.models.nn.utils import InferenceBatch
from etna.models.nn.utils import TrainBatch


class RNN(LightningModule):
    def __init__(self, input_size: int, num_layers: int = 2, hidden_size: int = 16, loss=nn.MSELoss()) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.loss = loss
        self.layer = nn.LSTM(
            num_layers=self.num_layers, hidden_size=self.hidden_size, input_size=self.input_size, batch_first=True
        )
        self.projection = nn.Linear(in_features=self.hidden_size, out_features=1)

    def forward(self, x: InferenceBatch):
        input_tensor = x["encoder_real"].float()  # (batch_size, encoder_length, input_size)
        decoder_real = x["decoder_real"].float()  # (batch_size, encoder_length, input_size)
        decoder_length = decoder_real.shape[1]
        output, (h_n, c_n) = self.layer(input_tensor)
        forecast = torch.zeros(size=(decoder_real.shape[0], decoder_real.shape[1], 1)).float().to(decoder_real.device)

        for i in range(decoder_length - 1):
            output, (h_n, c_n) = self.layer(decoder_real[:, i, None], (h_n, c_n))
            forecast_point = self.projection(output[:, -1]).flatten()
            forecast[:, i, 0] = forecast_point
            decoder_real[:, i + 1, 0] = forecast_point

        output, (h_n, c_n) = self.layer(decoder_real[:, decoder_length - 1, None], (h_n, c_n))
        forecast_point = self.projection(output[:, -1]).flatten()
        forecast[:, decoder_length - 1, 0] = forecast_point

        return forecast

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch: TrainBatch, batch_idx):
        encoder_real = train_batch["encoder_real"].float()  # (batch_size, encoder_lenght, input_size)
        decoder_real = train_batch["decoder_real"].float()  # (batch_size, decoder_length, input_size)
        target = train_batch["target"].float()

        decoder_length = decoder_real.shape[1]

        output, (_, _) = self.layer(torch.cat((encoder_real, decoder_real), dim=1))

        target_prediction = output[:, -decoder_length:]
        target_prediction = self.projection(target_prediction)

        # assert target_prediction.shape == target.shape

        target = target[:, -decoder_length:]

        return self.loss(target_prediction, target)
