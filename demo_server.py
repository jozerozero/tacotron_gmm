import falcon
import tensorflow as tf

from hparams import hparams
from infolog import log
from tacotron.synthesizer import Synthesizer
from wsgiref import simple_server
import argparse
import re
from pypinyin import pinyin, lazy_pinyin, Style


html_body = '''<html><title>Tcotron-2 Demo</title><meta charset='utf-8'>
<style>
body {padding: 16px; font-family: sans-serif; font-size: 14px; color: #444}
input {font-size: 14px; padding: 8px 12px; outline: none; border: 1px solid #ddd}
input:focus {box-shadow: 0 1px 2px rgba(0,0,0,.15)}
p {padding: 12px}
button {background: #28d; padding: 9px 14px; margin-left: 8px; border: none; outline: none;
        color: #fff; font-size: 14px; border-radius: 4px; cursor: pointer;}
button:hover {box-shadow: 0 1px 2px rgba(0,0,0,.15); opacity: 0.9;}
button:active {background: #29f;}
button[disabled] {opacity: 0.4; cursor: default}
</style>
<body>
<form>
  <input id="text" type="text" size="40" placeholder="请输入文字">
  <button id="button" name="synthesize">合成</button>
</form>
<p id="message"></p>
<audio id="audio" controls autoplay hidden></audio>
<script>
function q(selector) {return document.querySelector(selector)}
q('#text').focus()
q('#button').addEventListener('click', function(e) {
  text = q('#text').value.trim()
  if (text) {
    q('#message').textContent = '合成中...'
    q('#button').disabled = true
    q('#audio').hidden = true
    synthesize(text)
  }
  e.preventDefault()
  return false
})
function synthesize(text) {
  fetch('/synthesize?text=' + encodeURIComponent(text), {cache: 'no-cache'})
    .then(function(res) {
      if (!res.ok) throw Error(res.statusText)
      return res.blob()
    }).then(function(blob) {
      q('#message').textContent = ''
      q('#button').disabled = false
      q('#audio').src = URL.createObjectURL(blob)
      q('#audio').hidden = false
    }).catch(function(err) {
      q('#message').textContent = '出错: ' + err.message
      q('#button').disabled = false
    })
}
</script></body></html>
'''

def p(input):
	str = ""
	arr = pinyin(input, style=Style.TONE3)
	for i in arr:
		str += i[0] + " "
	return str

def replace_punc(text):
	return text.translate(text.maketrans("，。？：；！“”、（）",",.?:;!\"\",()"))

def remove_prosody(text):
	return re.sub(r'#[0-9]','',text)

parser = argparse.ArgumentParser()
#parser.add_argument('--checkpoint', default='pretrained/', help='Path to model checkpoint')
parser.add_argument('--checkpoint', default='logs-Tacotron/taco_pretrained/tacotron_model.ckpt-79000', help='Path to model checkpoint')
parser.add_argument('--hparams', default='',help='Hyperparameter overrides as a comma-separated list of name=value pairs')
parser.add_argument('--port', default=1234,help='Port of Http service')
parser.add_argument('--host', default="localhost",help='Host of Http service')
parser.add_argument('--name', help='Name of logging directory if the two models were trained together.')
args = parser.parse_args()
synth = Synthesizer()
modified_hp = hparams.parse(args.hparams)
synth.load(args.checkpoint, modified_hp)


class Res:
	def on_get(self,req,res):
		res.body = html_body
		res.content_type = "text/html"

class Syn:
	def on_get(self,req,res):
		print('IN')
		if not req.params.get('text'):
			raise falcon.HTTPBadRequest()
		res.body = p(remove_prosody(replace_punc(req.params.get('text'))))
		print(res.body)
		#synth.eval(res.data)
		synth.synthesize([res.body],None,None,None,None)
		res.content_type = "text/plain"		
		#res.data = synth.eval(p(req.params.get('text')))
		#res.content_type = "audio/wav"		
		

api = falcon.API()
api.add_route("/",Res())
api.add_route("/synthesize",Syn())
print("host:{},port:{}".format(args.host,int(args.port)))
simple_server.make_server(args.host,int(args.port),api).serve_forever()







