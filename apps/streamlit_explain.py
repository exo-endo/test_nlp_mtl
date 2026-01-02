import os
import streamlit as st
import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from captum.attr import IntegratedGradients

# Paths (resolve to absolute paths so transformers treats them as local dirs)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_CSV = os.path.join(BASE_DIR, 'Data', 'preprocessed.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'clinicalbert_homeless')
MODEL_NAME = MODEL_DIR  # load from local

# Device
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(device)
    return tokenizer, model

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_CSV)
    return df

def predict_and_attr(text, tokenizer, model, target_label=None, max_length=256):
    # Tokenize
    enc = tokenizer(text, truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')
    input_ids = enc['input_ids'].to(device)
    attention_mask = enc['attention_mask'].to(device)

    # compute prediction first (if needed)
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(out.logits, dim=-1).squeeze(0).cpu().numpy()

    pred_label = int(np.argmax(probs))
    if target_label is None:
        target_label = pred_label

    # embeddings
    embeddings = model.get_input_embeddings()(input_ids)

    def forward_func(inputs_embeds, attention_mask):
        outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        logits = outputs.logits
        return logits[:, target_label]

    ig = IntegratedGradients(forward_func)
    attributions, delta = ig.attribute(embeddings, additional_forward_args=(attention_mask,), return_convergence_delta=True)
    # collapse embedding dim
    token_attr = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()

    # get tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))

    return tokens, token_attr, probs, delta

def colorize(tokens, attributions, max_display=256):
    # Simple symmetric normalization and red↔white↔green mapping (no matplotlib needed)
    vals = np.array(attributions)
    # ignore special tokens when computing scale
    mask = np.array([t not in ['[CLS]', '[SEP]', '[PAD]'] for t in tokens])
    if mask.sum() == 0:
        scale = 1.0
    else:
        m = np.max(np.abs(vals[mask]))
        scale = m if m != 0 else 1.0

    def score_to_rgba(s):
        # s is in [-1,1]
        s = max(-1.0, min(1.0, s))
        alpha = 0.65
        if s >= 0:
            # interpolate white -> red
            r = 255
            g = int(255 * (1.0 - s))
            b = int(255 * (1.0 - s))
        else:
            # interpolate white -> green
            n = -s
            g = 255
            r = int(255 * (1.0 - n))
            b = int(255 * (1.0 - n))
        return f'rgba({r},{g},{b},{alpha})'

    spans = []
    for t, v in zip(tokens, vals):
        if t in ['[CLS]', '[SEP]', '[PAD]']:
            continue
        s = float(v) / scale
        color = score_to_rgba(s)
        # cleanup token display
        tok = t
        if tok.startswith('Ġ'):
            tok = tok[1:]
        tok = tok.replace('▁', '')
        tok = tok.replace('##', '')
        spans.append(f"<span style='background:{color}; padding:2px; margin:1px;border-radius:4px'>{tok}</span>")

    # build a CSS gradient legend (green -> white -> red) and numeric min/0/max
    minv = float(-scale)
    maxv = float(scale)
    legend_html = f"""
    <div style='width:100%; padding:6px;'>
        <div style='height:18px; background: linear-gradient(to right, rgba(0,200,0,0.65), rgba(255,255,255,0.8), rgba(255,0,0,0.65)); border-radius:4px;'></div>
        <div style='display:flex; justify-content:space-between; font-size:12px; color:#fff; margin-top:6px; font-weight:600; text-shadow: 0 1px 0 rgba(0,0,0,0.4);'>
            <div>{minv:.3f}</div>
            <div>0.000</div>
            <div>{maxv:.3f}</div>
        </div>
    </div>
    """

    return ' '.join(spans), legend_html

# Streamlit UI
st.set_page_config(page_title='ClinicalBERT SOCIAL HISTORY explain', layout='wide')
st.title('ClinicalBERT — SOCIAL HISTORY explain')

tokenizer, model = load_model_and_tokenizer()
df = load_data()

# Input source: dataset row or custom text
st.subheader('Input')
input_source = st.radio('Input source', ('Dataset row', 'Custom text'), index=0, horizontal=True)
if input_source == 'Dataset row':
    idx = st.number_input('Pick test row index (0..n-1)', min_value=0, max_value=len(df)-1, value=0)
    row = df.iloc[idx]
    # Use the canonical preprocessed columns if present, otherwise try fallbacks
    text = row.get('social_history_modified', None)
    if not text:
        text = row.get('text', None)
    if not text:
        # fallback to the first column value as string
        try:
            text = str(row.iloc[0])
        except Exception:
            text = ''
    true = row.get('sdoh_environment', row.get('homeless', None))
else:
    text = st.text_area('Enter custom SOCIAL HISTORY text to explain', value='', height=220)
    true = None

st.subheader('Original SOCIAL HISTORY text')
st.write(text if text else '_(empty)_')

col1, col2 = st.columns([3,1])
# create a placeholder in the legend column; we'll update it after computing attributions
with col2:
    st.markdown('**Attribution legend**')
    legend_placeholder = st.empty()
    legend_placeholder.markdown("<div style='color:#666'>Run 'Explain this note' to show a shaded legend matching the token colors.</div>", unsafe_allow_html=True)

with col1:
    if st.button('Explain this note'):
        if not text:
            st.warning('No text provided to explain.')
        else:
            with st.spinner('Computing attributions...'):
                tokens, token_attr, probs, delta = predict_and_attr(text, tokenizer, model)
                pred_label = int(np.argmax(probs))
                label_text = 'Positive (homeless)' if pred_label == 1 else 'Negative (not homeless)'
                st.markdown(f"**Label explained:** {label_text}")
                st.markdown(f"**Model probs (neg,pos):** {probs[0]:.3f}, {probs[1]:.3f}")
                html, legend_html = colorize(tokens, token_attr)
                st.markdown(html, unsafe_allow_html=True)
                # render the shaded legend in the right column
                if legend_html:
                    try:
                        legend_placeholder.markdown(legend_html, unsafe_allow_html=True)
                    except Exception:
                        # last-resort: replace with text
                        legend_placeholder.markdown('<div>Legend unavailable</div>', unsafe_allow_html=True)
                st.write('Convergence delta:', float(delta))
                if true is not None:
                    st.write('True label (homeless):', int(true))

st.markdown('---')
st.write("Notes: This uses IntegratedGradients over input embeddings. Attributions are computed toward the model's predicted label (shown above). Red indicates support for the predicted label; green indicates opposition.")
