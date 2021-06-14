import torch
import numpy as np


from cavity_model import (
    CavityModel,
    ResidueEnvironment,
    ResidueEnvironmentsDataset,
    ToTensor,
)


class Model_1(CavityModel):
    """
    Baseline model
    """
    def _model(self):
        self.xx, self.yy, self.zz = torch.tensor(
            np.meshgrid(
                self.lin_spacing_xy, self.lin_spacing_xy, self.lin_spacing_z, indexing="ij" # matrix indexing (classic python)
            ),
            dtype=torch.float32,
        ).to(self.device)

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv3d(self.n_atom_types, 16, kernel_size=(3, 3, 3), padding=1), # output = [100, 16, 4, 4 ,8]
            torch.nn.MaxPool3d(kernel_size=2),
            torch.nn.BatchNorm3d(16),
            torch.nn.ReLU(),
        )

        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=1), # usual: padding = round(kernel_size/2, lower), output = [100, 32, 2, 2, 4]
            torch.nn.MaxPool3d(kernel_size=2),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU(),
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1), # output = [100, 128, 1, 1, 2]
            torch.nn.MaxPool3d(kernel_size=2),
            torch.nn.BatchNorm3d(64),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
        )
        self.dense1 = torch.nn.Sequential(
            torch.nn.Linear(in_features=128, out_features=256), # bachnorm filters 64 * 4 parameters of batch norm per filter
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
        )
        self.dense2 = torch.nn.Linear(in_features=256, out_features=40)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._gaussian_blurring(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.dense1(x)
        x = self.dense2(x) # yields logits, mapped to probabilities afterwards by the Softmax fct.
        return x


class Model_2(CavityModel):
    """
    Model_1 with half the number of conv filters and dense nodes.
    """
    def _model(self):
        self.xx, self.yy, self.zz = torch.tensor(
            np.meshgrid(
                self.lin_spacing_xy, self.lin_spacing_xy, self.lin_spacing_z, indexing="ij" # matrix indexing (classic python)
            ),
            dtype=torch.float32,
        ).to(self.device)

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv3d(self.n_atom_types, 8, kernel_size=(3, 3, 3), padding=1), # output = [100, 16, 4, 4 ,8]
            torch.nn.MaxPool3d(kernel_size=2),
            torch.nn.BatchNorm3d(8),
            torch.nn.ReLU(),
        )

        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv3d(8, 16, kernel_size=(3, 3, 3), padding=1), # usual: padding = round(kernel_size/2, lower), output = [100, 32, 2, 2, 4]
            torch.nn.MaxPool3d(kernel_size=2),
            torch.nn.BatchNorm3d(16),
            torch.nn.ReLU(),
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=1), # output = [100, 128, 1, 1, 2]
            torch.nn.MaxPool3d(kernel_size=2),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
        )
        self.dense1 = torch.nn.Sequential(
            torch.nn.Linear(in_features=64, out_features=128), # bachnorm filters 64 * 4 parameters of batch norm per filter
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
        )
        self.dense2 = torch.nn.Linear(in_features=128, out_features=40)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._gaussian_blurring(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.dense1(x)
        x = self.dense2(x) # yields logits, mapped to probabilities afterwards by the Softmax fct.
        return x


class Model_3(CavityModel):
    """
    Model_1 with less conv filters.
    """
    def _model(self):
        self.xx, self.yy, self.zz = torch.tensor(
            np.meshgrid(
                self.lin_spacing_xy, self.lin_spacing_xy, self.lin_spacing_z, indexing="ij" # matrix indexing (classic python)
            ),
            dtype=torch.float32,
        ).to(self.device)

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv3d(self.n_atom_types, 8, kernel_size=(3, 3, 3), padding=1), # output = [100, 16, 4, 4 ,8]
            torch.nn.MaxPool3d(kernel_size=2),
            torch.nn.BatchNorm3d(8),
            torch.nn.ReLU(),
        )

        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv3d(8, 16, kernel_size=(3, 3, 3), padding=1), # usual: padding = round(kernel_size/2, lower), output = [100, 32, 2, 2, 4]
            torch.nn.MaxPool3d(kernel_size=2),
            torch.nn.BatchNorm3d(16),
            torch.nn.ReLU(),
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=1), # output = [100, 128, 1, 1, 2]
            torch.nn.MaxPool3d(kernel_size=2),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
        )
        self.dense1 = torch.nn.Sequential(
            torch.nn.Linear(in_features=64, out_features=256), # bachnorm filters 64 * 4 parameters of batch norm per filter
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
        )
        self.dense2 = torch.nn.Linear(in_features=256, out_features=40)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._gaussian_blurring(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.dense1(x)
        x = self.dense2(x) # yields logits, mapped to probabilities afterwards by the Softmax fct.
        return x


class Model_5(CavityModel):
    """
    Change the size of the last conv layer
    """
    def _model(self):
        self.xx, self.yy, self.zz = torch.tensor(
            np.meshgrid(
                self.lin_spacing_xy, self.lin_spacing_xy, self.lin_spacing_z, indexing="ij" # matrix indexing (classic python)
            ),
            dtype=torch.float32,
        ).to(self.device)

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv3d(self.n_atom_types, 16, kernel_size=(3, 3, 3), padding=1), # output = [100, 16, 4, 4 ,8]
            torch.nn.MaxPool3d(kernel_size=2),
            torch.nn.BatchNorm3d(16),
            torch.nn.ReLU(),
        )

        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=1), # usual: padding = round(kernel_size/2, lower), output = [100, 32, 2, 2, 4]
            torch.nn.MaxPool3d(kernel_size=2),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU(),
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1), # output = [100, 128, 1, 1, 2]
            torch.nn.BatchNorm3d(64),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
        )
        self.dense1 = torch.nn.Sequential(
            torch.nn.Linear(in_features=1024, out_features=256), # bachnorm filters 64 * 4 parameters of batch norm per filter
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
        )
        self.dense2 = torch.nn.Linear(in_features=256, out_features=40)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._gaussian_blurring(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.dense1(x)
        x = self.dense2(x) # yields logits, mapped to probabilities afterwards by the Softmax fct.
        return x



class Model_8(CavityModel):
    """
    Similar to 5 but with larger last dense layer.
    """
    def _model(self):
        self.xx, self.yy, self.zz = torch.tensor(
            np.meshgrid(
                self.lin_spacing_xy, self.lin_spacing_xy, self.lin_spacing_z, indexing="ij" # matrix indexing (classic python)
            ),
            dtype=torch.float32,
        ).to(self.device)

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv3d(self.n_atom_types, 16, kernel_size=(3, 3, 3), padding=1), # output = [100, 16, 4, 4 ,8]
            torch.nn.MaxPool3d(kernel_size=2),
            torch.nn.BatchNorm3d(16),
            torch.nn.ReLU(),
        )

        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=1), # usual: padding = round(kernel_size/2, lower), output = [100, 32, 2, 2, 4]
            torch.nn.MaxPool3d(kernel_size=2),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU(),
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1), # output = [100, 128, 1, 1, 2]
            torch.nn.BatchNorm3d(64),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
        )
        self.dense1 = torch.nn.Sequential(
            torch.nn.Linear(in_features=1024, out_features=512), # bachnorm filters 64 * 4 parameters of batch norm per filter
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
        )
        self.dense2 = torch.nn.Linear(in_features=512, out_features=40)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._gaussian_blurring(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.dense1(x)
        x = self.dense2(x) # yields logits, mapped to probabilities afterwards by the Softmax fct.
        return x


class Model_9(CavityModel):
    """
    Similar to 5 and 8 but with larger last dense layer. Differs from 8 by smaller last dense layer.
    """
    def _model(self):
        self.xx, self.yy, self.zz = torch.tensor(
            np.meshgrid(
                self.lin_spacing_xy, self.lin_spacing_xy, self.lin_spacing_z, indexing="ij" # matrix indexing (classic python)
            ),
            dtype=torch.float32,
        ).to(self.device)

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv3d(self.n_atom_types, 16, kernel_size=(3, 3, 3), padding=1), # output = [100, 16, 4, 4 ,8]
            torch.nn.MaxPool3d(kernel_size=2),
            torch.nn.BatchNorm3d(16),
            torch.nn.ReLU(),
        )

        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=1), # usual: padding = round(kernel_size/2, lower), output = [100, 32, 2, 2, 4]
            torch.nn.MaxPool3d(kernel_size=2),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU(),
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1), # output = [100, 128, 1, 1, 2]
            torch.nn.BatchNorm3d(64),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
        )
        self.dense1 = torch.nn.Sequential(
            torch.nn.Linear(in_features=1024, out_features=128), # bachnorm filters 64 * 4 parameters of batch norm per filter
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
        )
        self.dense2 = torch.nn.Linear(in_features=128, out_features=40)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._gaussian_blurring(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.dense1(x)
        x = self.dense2(x) # yields logits, mapped to probabilities afterwards by the Softmax fct.
        return x


class Model_ae_1(CavityModel):
    """
    Baseline model
    """
    def _model(self):
        self.xx, self.yy, self.zz = torch.tensor(
            np.meshgrid(
                self.lin_spacing_xy, self.lin_spacing_xy, self.lin_spacing_z, indexing="ij" # matrix indexing (classic python)
            ),
            dtype=torch.float32,
        ).to(self.device)

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv3d(self.n_atom_types, 16, kernel_size=(3, 3, 3), padding=1), # output = [100, 16, 4, 4 ,8]
            torch.nn.MaxPool3d(kernel_size=2),
            torch.nn.BatchNorm3d(16),
            torch.nn.ReLU(),
        )

        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=1), # usual: padding = round(kernel_size/2, lower), output = [100, 32, 2, 2, 4]
            torch.nn.MaxPool3d(kernel_size=2),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU(),
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1), # output = [100, 128, 1, 1, 2]
            torch.nn.MaxPool3d(kernel_size=2),
            torch.nn.BatchNorm3d(64),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
        )
        self.dense1 = torch.nn.Sequential(
            torch.nn.Linear(in_features=128, out_features=256), # bachnorm filters 64 * 4 parameters of batch norm per filter
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
        )
        self.dense2 = torch.nn.Linear(in_features=256, out_features=120)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._gaussian_blurring(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.dense1(x)
        x = self.dense2(x) # yields logits, mapped to probabilities afterwards by the Softmax fct.
        return x


class Model_matfact_1(CavityModel):
    """
    Similar to 5 and 8 but with larger last dense layer. Differs from 8 by smaller last dense layer.
    """
    def _model(self):
        self.xx, self.yy, self.zz = torch.tensor(
            np.meshgrid(
                self.lin_spacing_xy, self.lin_spacing_xy, self.lin_spacing_z, indexing="ij" # matrix indexing (classic python)
            ),
            dtype=torch.float32,
        ).to(self.device)

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv3d(self.n_atom_types, 16, kernel_size=(3, 3, 3), padding=1), # output = [100, 16, 4, 4 ,8]
            torch.nn.MaxPool3d(kernel_size=2),
            torch.nn.BatchNorm3d(16),
            torch.nn.ReLU(),
        )

        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=1), # usual: padding = round(kernel_size/2, lower), output = [100, 32, 2, 2, 4]
            torch.nn.MaxPool3d(kernel_size=2),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU(),
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1), # output = [100, 128, 1, 1, 2]
            torch.nn.BatchNorm3d(64),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
        )
        self.dense1 = torch.nn.Sequential(
            torch.nn.Linear(in_features=1024, out_features=256), # bachnorm filters 64 * 4 parameters of batch norm per filter
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
        )
        self.dense2 = torch.nn.Linear(in_features=256, out_features=120)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._gaussian_blurring(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.dense1(x)
        x = self.dense2(x) # yields logits, mapped to probabilities afterwards by the Softmax fct.
        return x


class Model_matfact_1_full(CavityModel):
    """
    For the full size of the grid (18x18x18)
    """
    def _model(self):
        self.xx, self.yy, self.zz = torch.tensor(
            np.meshgrid(
                self.lin_spacing_xy, self.lin_spacing_xy, self.lin_spacing_z, indexing="ij" # matrix indexing (classic python)
            ),
            dtype=torch.float32,
        ).to(self.device)

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv3d(self.n_atom_types, 16, kernel_size=(3, 3, 3), padding=1), # output = [100, 16, 8, 8 ,8]
            torch.nn.MaxPool3d(kernel_size=2),
            torch.nn.BatchNorm3d(16),
            torch.nn.ReLU(),
        )

        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=1), # usual: padding = round(kernel_size/2, lower), output = [100, 32, 4, 4, 4]
            torch.nn.MaxPool3d(kernel_size=2),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU(),
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1), # output = [100, 128, 2, 2, 2]
            torch.nn.BatchNorm3d(64),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
        )
        self.dense1 = torch.nn.Sequential(
            torch.nn.Linear(in_features=4096, out_features=256), # bachnorm filters 64 * 4 parameters of batch norm per filter
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
        )
        self.dense2 = torch.nn.Linear(in_features=256, out_features=120)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._gaussian_blurring(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.dense1(x)
        x = self.dense2(x) # yields logits, mapped to probabilities afterwards by the Softmax fct.
        return x


class Model_matfact_2(CavityModel):
    """
    Similar to 5 and 8 but with larger last dense layer. Differs from 8 by smaller last dense layer.
    """
    def _model(self):
        self.xx, self.yy, self.zz = torch.tensor(
            np.meshgrid(
                self.lin_spacing_xy, self.lin_spacing_xy, self.lin_spacing_z, indexing="ij" # matrix indexing (classic python)
            ),
            dtype=torch.float32,
        ).to(self.device)

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv3d(self.n_atom_types, 16, kernel_size=(3, 3, 3), padding=1), # output = [100, 16, 4, 4 ,8]
            torch.nn.MaxPool3d(kernel_size=2),
            torch.nn.BatchNorm3d(16),
            torch.nn.ReLU(),
        )

        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=1), # usual: padding = round(kernel_size/2, lower), output = [100, 32, 2, 2, 4]
            torch.nn.MaxPool3d(kernel_size=2),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU(),
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1), # output = [100, 128, 1, 1, 2]
            torch.nn.BatchNorm3d(64),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
        )
        self.dense1 = torch.nn.Sequential(
            torch.nn.Linear(in_features=1024, out_features=256), # bachnorm filters 64 * 4 parameters of batch norm per filter
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
        )
        self.dense2 = torch.nn.Linear(in_features=256, out_features=80)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._gaussian_blurring(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.dense1(x)
        x = self.dense2(x) # yields logits, mapped to probabilities afterwards by the Softmax fct.
        return x


class Model_matfact_3(CavityModel):
    """
    Similar to 5 and 8 but with larger last dense layer. Differs from 8 by smaller last dense layer.
    """
    def _model(self):
        self.xx, self.yy, self.zz = torch.tensor(
            np.meshgrid(
                self.lin_spacing_xy, self.lin_spacing_xy, self.lin_spacing_z, indexing="ij" # matrix indexing (classic python)
            ),
            dtype=torch.float32,
        ).to(self.device)

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv3d(self.n_atom_types, 16, kernel_size=(3, 3, 3), padding=1), # output = [100, 16, 4, 4 ,8]
            torch.nn.MaxPool3d(kernel_size=2),
            torch.nn.BatchNorm3d(16),
            torch.nn.ReLU(),
        )

        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=1), # usual: padding = round(kernel_size/2, lower), output = [100, 32, 2, 2, 4]
            torch.nn.MaxPool3d(kernel_size=2),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU(),
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1), # output = [100, 128, 1, 1, 2]
            torch.nn.BatchNorm3d(64),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
        )
        self.dense1 = torch.nn.Sequential(
            torch.nn.Linear(in_features=1024, out_features=512), # bachnorm filters 64 * 4 parameters of batch norm per filter
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
        )
        self.dense2 = torch.nn.Linear(in_features=512, out_features=160)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._gaussian_blurring(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.dense1(x)
        x = self.dense2(x) # yields logits, mapped to probabilities afterwards by the Softmax fct.
        return x


class Model_matfact_4(CavityModel):
    """
    Similar to 5 and 8 but with larger last dense layer. Differs from 8 by smaller last dense layer.
    """
    def _model(self):
        self.xx, self.yy, self.zz = torch.tensor(
            np.meshgrid(
                self.lin_spacing_xy, self.lin_spacing_xy, self.lin_spacing_z, indexing="ij" # matrix indexing (classic python)
            ),
            dtype=torch.float32,
        ).to(self.device)

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv3d(self.n_atom_types, 16, kernel_size=(3, 3, 3), padding=1), # output = [100, 16, 4, 4 ,8]
            torch.nn.MaxPool3d(kernel_size=2),
            torch.nn.BatchNorm3d(16),
            torch.nn.ReLU(),
        )

        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=1), # usual: padding = round(kernel_size/2, lower), output = [100, 32, 2, 2, 4]
            torch.nn.MaxPool3d(kernel_size=2),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU(),
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1), # output = [100, 128, 1, 1, 2]
            torch.nn.BatchNorm3d(64),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
        )
        self.dense1 = torch.nn.Sequential(
            torch.nn.Linear(in_features=1024, out_features=256), # bachnorm filters 64 * 4 parameters of batch norm per filter
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
        )
        self.dense2 = torch.nn.Linear(in_features=256, out_features=160)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._gaussian_blurring(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.dense1(x)
        x = self.dense2(x) # yields logits, mapped to probabilities afterwards by the Softmax fct.
        return x