import re
import string
import json
import argparse
import os
import torch
from pytorch_pretrained_bert import BertConfig


from models import data_loader, model_builder
from models.data_loader import load_dataset
from models.model_builder import Summarizer
from models.trainer import build_trainer
from others.logging import logger, init_logger


from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

from flask_cors import CORS

CORS(app)

model_flags = ['hidden_size', 'ff_size', 'heads', 'inter_layers','encoder','ff_actv', 'use_interval','rnn_size']

translator = None

parser = argparse.ArgumentParser()
args = parser.parse_args()
args.oracle_mode = 'greedy'
args.test_from = '../models//bert_classifier/cnndm_bertsum_classifier_best.pt'
args.bert_config_path = '../bert_config_uncased_base.json'
args.temp_dir = '../temp'
args.param_init = 0.0
args.param_init_glorot = True
args.visible_gpus = -1
args.accum_count = 1
args.world_size = 1
args.model_path = '../models/'
args.report_every = 5000
args.save_checkpoint_steps = 5000
args.min_src_ntokens=5
args.max_src_ntokens=200
args.min_nsents=3
args.max_nsents=100
args.block_trigram=True
args.lower=True

def getTranslator():
    # set up model

    device = "cpu"

    logger.info('Loading checkpoint from %s' % args.test_from)
    checkpoint = torch.load(args.test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])

    print(args)

    config = BertConfig.from_json_file(args.bert_config_path)
    model = Summarizer(args, device, load_pretrained_bert=False, bert_config=config)
    model.load_cp(checkpoint)
    model.eval()

    return  build_trainer(args, -1, model, None)


@app.route('/extract', methods=['GET', 'POST'])
def translate():
    if request.method == 'POST':
        # user inputs
        req = request.json
        print(req)

        # api call
        res={}
        if len(req['src']) >0:
            print("request src "+str(req['src']))

            #global translator
            #if translator is None:
            #    print("loading model")
            #    translator=getTranslator()
            #    print("model loaded")
            # res=translator.translate(args, req['src'])

            try:
                with open('src_text.txt','w') as f:
                    f.write(req['src'])

                os.system('./translate.sh')
                with open('result.json') as json_file:
                    res = json.load(json_file)
            except:
                print("process error, return empty str")

        return jsonify(res)


    return render_template('index.html')

# HTTP Errors handlers
@app.errorhandler(404)
def url_error(e):
    return """
    Wrong URL!
    <pre>{}</pre>""".format(e), 404


@app.errorhandler(500)
def server_error(e):
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8081))
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader=True)
    app.run(debug=True)
