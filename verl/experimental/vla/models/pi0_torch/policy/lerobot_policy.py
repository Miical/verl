# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing_extensions import override

from .base import Pi0Output
from .libero_policy import LiberoPi0Input

PI0_MAX_STATE_DIM = 32
PI0_ACTION_CHUNK_SIZE = 10
LEROBOT_ACTION_DIM = 6


class LerobotPi0Input(LiberoPi0Input):
    ...

class LerobotPi0Output(Pi0Output):
    @override
    @classmethod
    def from_model_output(cls, model_output: dict) -> "LerobotPi0Output":
        output = cls()
        output.action = model_output["full_action"][:, :PI0_ACTION_CHUNK_SIZE, :LEROBOT_ACTION_DIM]
        output.log_prob = model_output["log_probs"]
        return output
