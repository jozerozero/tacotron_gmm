import sys
import os
import numpy as np
from tacotron.utils import plot

lin_p=np.load(sys.argv[1])
plot.plot_spectrogram(lin_p,
        os.path.join('', 'tmp-training-linear-spectrogram.png'),
        #title='{}, {}, step={}, loss={:.5f}'.format(args.model, time_string(), step, eval_loss), 
        #target_spectrogram=lin_t,
        #max_len=t_len, 
        auto_aspect=True)
