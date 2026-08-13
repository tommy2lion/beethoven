"""
Beethoven - Music Source Separation Web App
=============================================
Three modes: 4-stem, 6-stem (Demucs), and 构想二 (Generative + MIDI)
"""

import gradio as gr
import subprocess, sys, shutil, os
from pathlib import Path
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

# ========== Standalone / Frozen Environment ==========
def get_base_dir():
    """Get the project base directory, works in both normal and frozen (PyInstaller) modes."""
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).parent.parent
    return Path(__file__).parent.parent

def get_samples_dir():
    """Get the samples directory, writable location for separated audio."""
    if getattr(sys, 'frozen', False):
        return get_base_dir() / "samples"
    return get_base_dir() / "samples"

DEVICE = "cuda" if __import__('torch').cuda.is_available() else "cpu"
SR = 22050
N_FFT = 2048
HOP_LENGTH = 256
PROJECT_DIR = get_base_dir()
OUTPUT_BASE = get_samples_dir() / "separated" / "webapp"
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

# ========== Demucs Direct API (for standalone .exe) ==========
DEMUCS_AVAILABLE = False
DEMUCS_SOURCES = {'htdemucs': ['drums', 'bass', 'other', 'vocals'],
                  'htdemucs_6s': ['drums', 'bass', 'other', 'vocals', 'guitar', 'piano']}
DEMUCS_SAMPLERATE = 44100

def _run_demucs_api(model_name, output_dir, audio_path):
    """Run Demucs separation using direct Python API (not subprocess)."""
    from demucs.api import Separator, save_audio
    import torch as _th

    separator = Separator(
        model=model_name,
        device=DEVICE if _th.cuda.is_available() else "cpu",
        jobs=0,
        progress=False,
    )
    origin, res = separator.separate_audio_file(str(audio_path))
    # res shape: (sources, channels, samples)
    out_dir = Path(output_dir) / model_name / Path(audio_path).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, audio in zip(separator.model.sources, res):
        save_audio(audio, str(out_dir / f"{name}.wav"), samplerate=separator.samplerate)
    return out_dir

def _run_demucs(model_name, output_dir, audio_path):
    """Run Demucs separation. Uses direct API when frozen (PyInstaller), falls back to subprocess."""
    out_dir = Path(output_dir) / model_name / Path(audio_path).stem
    if getattr(sys, 'frozen', False):
        return _run_demucs_api(model_name, output_dir, audio_path)
    # Normal mode: use subprocess
    result = subprocess.run(
        [sys.executable, "-m", "demucs", "-n", model_name, "-o", str(output_dir), str(audio_path)],
        capture_output=True, text=True, timeout=600
    )
    if result.returncode != 0:
        raise RuntimeError(f"Demucs failed: {result.stderr[:500]}")
    return out_dir

# ========== Model Configs ==========
MODELS = {
    '4-stem (人声/贝斯/鼓/其他)': {
        'name': 'htdemucs',
        'mode': 'demucs',
        'instruments': ['vocals', 'bass', 'drums', 'other'],
    },
    '6-stem (+吉他/钢琴)': {
        'name': 'htdemucs_6s',
        'mode': 'demucs',
        'instruments': ['vocals', 'bass', 'drums', 'guitar', 'piano', 'other'],
    },
    '构想二 (生成式 + MIDI)': {
        'name': 'approach2',
        'mode': 'generative',
        'instruments': ['vocals', 'bass', 'drums', 'guitar', 'piano', 'other'],
    },
}

COLORS = { 'vocals':'#e74c3c','bass':'#2ecc71','drums':'#f39c12',
           'guitar':'#9b59b6','piano':'#1abc9c','other':'#3498db' }
ICONS = { 'vocals':'🎤','bass':'🎸','drums':'🥁',
          'guitar':'🎸','piano':'🎹','other':'🎹' }
INSTRUMENT_NAMES = {
    'vocals':{'zh':'人声','en':'Vocals'}, 'bass':{'zh':'贝斯','en':'Bass'},
    'drums':{'zh':'鼓','en':'Drums'}, 'guitar':{'zh':'吉他','en':'Guitar'},
    'piano':{'zh':'钢琴','en':'Piano'}, 'other':{'zh':'其他','en':'Other'},
}
ALL_INSTRUMENTS = ['vocals', 'bass', 'drums', 'guitar', 'piano', 'other']

# ========== Instrument Synthesizer (for Approach 2) ==========
INST_PARAMS = {
    'piano':{'harm':[1,2,3,4,5,6,8],'wt':[1,.6,.3,.15,.08,.04,.02],'decay':5},
    'guitar':{'harm':[1,2,3,4,5],'wt':[1,.4,.2,.1,.05],'decay':4},
    'bass':{'harm':[1,3],'wt':[1,.2],'decay':3},
    'vocals':{'harm':[1,2,3,4,5],'wt':[1,.3,.1,.05,.02],'decay':2,'vib':True},
    'drums':{'harm':[1,2,3],'wt':[1,.5,.25],'decay':20},
    'violin':{'harm':[1,2,3,4],'wt':[1,.5,.2,.08],'decay':1.2,'vib':True},
}

def synth_note(instr, freq, dur, sr=SR):
    t = np.linspace(0, dur, int(sr*dur), endpoint=False)
    if len(t)==0: return np.array([])
    p = INST_PARAMS.get(instr, INST_PARAMS['piano'])
    if p.get('vib'):
        vib = 0.05*np.sin(2*np.pi*5.5*t)
        ph = 2*np.pi*np.cumsum(freq*(1+vib))/sr
        note = sum(w*np.sin(h*ph) for h,w in zip(p['harm'],p['wt']))
    else:
        note = sum(w*np.sin(2*np.pi*freq*h*t) for h,w in zip(p['harm'],p['wt']))
    atk = np.minimum(1.0, t/0.02) if instr!='piano' else 1.0
    env = atk * np.exp(-t*p['decay'])
    if instr=='drums': env += np.random.randn(len(t))*0.01
    note = note * env
    pk = np.max(np.abs(note))
    return note/pk*0.3 if pk>0 else note

# ========== Approach 2 Pipeline ==========
def run_approach2(audio_path):
    """Run the full generative pipeline. Returns (output_dir, notes_summary)."""
    # Ensure Demucs 6s stems exist
    out_root = PROJECT_DIR / "samples" / "separated"
    demucs_dir = out_root / "htdemucs_6s" / Path(audio_path).stem
    if not demucs_dir.exists():
        _run_demucs("htdemucs_6s", out_root, audio_path)

    approach2_dir = out_root / "approach2"
    midi_dir = approach2_dir / "midi"
    approach2_dir.mkdir(parents=True, exist_ok=True)
    midi_dir.mkdir(exist_ok=True)

    y_mix, _ = librosa.load(audio_path, sr=SR)
    duration = len(y_mix) / SR

    summary = {}
    for inst in ['vocals','bass','drums','guitar','piano','other']:
        stem_path = demucs_dir / f"{inst}.wav"
        if not stem_path.exists():
            summary[inst] = {'notes': 0, 'rms': 0}
            continue

        # Note detection via spectral peaks
        y_stem, sr = librosa.load(stem_path, sr=SR)
        D = np.abs(librosa.stft(y_stem, n_fft=2048, hop_length=HOP_LENGTH))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        times = librosa.times_like(D, sr=sr, hop_length=HOP_LENGTH)

        onsets = librosa.onset.onset_detect(y=y_stem, sr=sr, hop_length=HOP_LENGTH, backtrack=True)
        ot = librosa.frames_to_time(onsets, sr=sr, hop_length=HOP_LENGTH)
        if len(ot) < 2:
            ot = times[::max(1,int(0.2*sr/HOP_LENGTH))]

        notes = []
        fb = max(0, np.argmax(freqs>=60)-1)
        fe = min(len(freqs), np.argmax(freqs>=2000)+1)
        for i in range(len(ot)):
            sf = max(0, int(ot[i]*sr/HOP_LENGTH))
            ef = min(D.shape[1], int((ot[i+1] if i+1<len(ot) else times[-1])*sr/HOP_LENGTH)+1)
            if ef-sf < 1: continue
            seg = np.mean(D[fb:fe,sf:ef], axis=1)
            peaks, props = find_peaks(seg, height=np.max(seg)*0.1, distance=3, prominence=0.02)
            if len(peaks):
                best = peaks[np.argmax(props['peak_heights'])]
                pitch = freqs[fb+best]
                conf = props['peak_heights'][np.argmax(props['peak_heights'])]/(np.max(seg)+1e-10)
                if pitch>60: notes.append((ot[i], ot[i+1] if i+1<len(ot) else times[-1], pitch, min(1,conf)))

        # Group consecutive similar notes
        if notes:
            grouped = [notes[0]]
            for n in notes[1:]:
                if abs(n[2]-grouped[-1][2])/grouped[-1][2] < 0.05 and n[0]-grouped[-1][1] < 0.05:
                    grouped[-1] = (grouped[-1][0], n[1], grouped[-1][2], max(grouped[-1][3],n[3]))
                else:
                    grouped.append(n)
            notes = grouped

        # Resynthesize
        n_s = int(duration * SR)
        track = np.zeros(n_s)
        for start, end, pitch, conf in notes:
            d = min(end-start, 3.0)
            if d < 0.05: continue
            note = synth_note(inst, pitch, d)
            ss = int(start*SR)
            note = note[:min(len(note), n_s-ss)]
            if len(note): track[ss:ss+len(note)] += note * conf * 0.8

        # Confidence filter
        ws = int(0.1 * SR)
        filt = np.zeros(n_s)
        for s in range(0, n_s-ws, ws//2):
            e = min(s+ws, n_s)
            rt = np.sqrt(np.mean(track[s:e]**2)+1e-10)
            ro = np.sqrt(np.mean(y_mix[s:e]**2)+1e-10)
            if rt>0.001 and ro>0.001:
                corr = np.sum(track[s:e]*y_mix[s:e])/(rt*ro*len(track[s:e]))
                if corr*min(rt/ro,ro/rt) > 0.15:
                    filt[s:e] = track[s:e]

        # Save
        import soundfile as _sf
        _sf.write(approach2_dir/f"{inst}.wav", filt, SR)
        _sf.write(approach2_dir/f"{inst}_raw.wav", track, SR)

        # MIDI
        try:
            from midiutil import MIDIFile
            gm = {'piano':0,'guitar':25,'bass':34,'violin':41,'vocals':54,'drums':0,'other':1}
            mf = MIDIFile(1, deinterleave=False)
            mf.addTrackName(0, 0, inst)
            mf.addTempo(0, 0, 120)
            mf.addProgramChange(0, 0, 0, gm.get(inst, 0))
            for start, end, pitch, conf in notes:
                mn = max(0, min(127, int(round(librosa.hz_to_midi(pitch)))))
                if mn > 0:
                    try:
                        mf.addNote(0, 0, mn, max(0,start), max(0.05,end-start), min(127,int(conf*80+30)))
                    except: pass
            with open(midi_dir/f"{inst}.mid", 'wb') as f:
                mf.writeFile(f)
        except: pass

        summary[inst] = {'notes': len(notes), 'rms': float(np.sqrt(np.mean(filt**2)))}

    return approach2_dir, summary


# ========== i18n ==========
LANG = {
    'title':{'zh':'🎵 Beethoven - 音乐多乐器声源分离','en':'🎵 Beethoven - Music Source Separation'},
    'upload':{'zh':'📁 上传音频','en':'📁 Upload Audio'},
    'model':{'zh':'🧩 分离模式','en':'🧩 Model'},
    'separate':{'zh':'🎧 开始分离','en':'🎧 Separate'},
    'separating':{'zh':'⏳ 分离中...','en':'⏳ Separating...'},
    'original':{'zh':'原始混合','en':'Original Mix'},
    'spec':{'zh':'频谱对比图','en':'Spectrogram'},
    'status':{'zh':'📋 状态','en':'📋 Status'},
    'upload_first':{'zh':'⚠️ 请先上传音频','en':'⚠️ Upload an audio file first'},
    'success':{'zh':'✅ 分离完成','en':'✅ Separation complete!'},
    'duration':{'zh':'时长','en':'Duration'},
    'notes_found':{'zh':'检测到','en':'Notes detected'},
    'midi_files':{'zh':'🎵 MIDI 乐谱已生成','en':'🎵 MIDI files generated'},
    'footer':{'zh':'---\n**Beethoven** — 构想一(掩码法) + 构想二(生成式+MIDI)','en':'---\n**Beethoven** — Approach 1 (Masking) + Approach 2 (Generative+MIDI)'},
}

def _(key, lang='zh'): return LANG.get(key,{}).get(lang,key)
def iname(inst, lang='zh'): return f"{ICONS.get(inst,'')} {INSTRUMENT_NAMES.get(inst,{}).get(lang,inst)}"

# ========== Processing ==========
def plot_specs(input_path, out_dir, instruments, lang='zh'):
    n = len(instruments)+1
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.5*n))
    y,_ = librosa.load(input_path, sr=SR)
    D = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mag = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    librosa.display.specshow(mag, sr=SR, hop_length=HOP_LENGTH, x_axis='s', y_axis='hz', ax=axes[0], cmap='magma', vmin=-50, vmax=0)
    axes[0].set_title(_('original',lang), fontsize=13, fontweight='bold')
    axes[0].set_ylim(0,8000)
    for i,inst in enumerate(instruments):
        p = out_dir/f"{inst}.wav"
        if p.exists():
            ys,_ = librosa.load(p, sr=SR)
            Ds = librosa.stft(ys, n_fft=N_FFT, hop_length=HOP_LENGTH)
            ms = librosa.amplitude_to_db(np.abs(Ds), ref=max(np.max(np.abs(Ds)),1))
            librosa.display.specshow(ms, sr=SR, hop_length=HOP_LENGTH, x_axis='s', y_axis='hz', ax=axes[i+1], cmap='magma', vmin=-50, vmax=0)
            axes[i+1].set_title(iname(inst,lang), fontsize=12, fontweight='bold', color=COLORS.get(inst,'white'))
            axes[i+1].set_ylim(0,8000)
    plt.tight_layout()
    p = OUTPUT_BASE / f"{Path(input_path).stem}_spec.png"
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return str(p)

def process(file_obj, model_key, lang):
    if file_obj is None:
        return [None]*9 + [_('upload_first',lang)]

    cfg = MODELS[model_key]
    instruments = cfg['instruments']
    input_path = file_obj.name
    stem_name = Path(input_path).stem

    try:
        if cfg['mode'] == 'generative':
            # Approach 2
            yield [None]*9 + [_('separating',lang)]
            out_dir, summary = run_approach2(input_path)
            spec_path = plot_specs(input_path, out_dir, instruments, lang)

            midi_dir = out_dir / "midi"
            midi_found = list(midi_dir.glob("*.mid")) if midi_dir.exists() else []

            total_notes = sum(s['notes'] for s in summary.values())
            info = f"{_('success',lang)} | {_('notes_found',lang)}: {total_notes}\n"
            for inst in instruments:
                s = summary.get(inst, {'notes':0,'rms':0})
                info += f"  {iname(inst,lang)}: {s['notes']} 音符, RMS={s['rms']:.4f}\n"
            if midi_found:
                info += f"\n{_('midi_files',lang)}: {', '.join(m.name for m in midi_found)}"
        else:
            # Demucs - use direct API (frozen) or subprocess (normal)
            yield [None]*9 + [_('separating',lang)]
            out_dir = _run_demucs(cfg['name'], OUTPUT_BASE, input_path)
            spec_path = plot_specs(input_path, out_dir, instruments, lang)

            y,_ = librosa.load(input_path, sr=SR)
            info = f"{_('success',lang)}\n  {Path(input_path).name}\n  {_('duration',lang)}: {len(y)/SR:.0f}s\n  Model: {cfg['name']}"

        # Build audio outputs
        audio_out = {}
        for inst in ALL_INSTRUMENTS:
            p = out_dir / f"{inst}.wav"
            audio_out[inst] = str(p) if p.exists() else None

        yield [spec_path, str(input_path),
               audio_out.get('vocals'), audio_out.get('bass'), audio_out.get('drums'),
               audio_out.get('guitar'), audio_out.get('piano'), audio_out.get('other'),
               info]

    except Exception as e:
        yield [None]*9 + [f"❌ Error: {str(e)}"]


# ========== UI ==========
with gr.Blocks(title="Beethoven - Music Source Separation") as app:
    gr.HTML("""
    <style>
        footer {display:none!important}
        .model-desc {font-size:13px; color:#666; margin: -8px 0 12px 20px}
    </style>
    """)

    with gr.Row():
        gr.Column(scale=4)
        with gr.Column(scale=1, min_width=200):
            lang_radio = gr.Radio([("🇨🇳 中文","zh"),("🇬🇧 English","en")], value="zh", label="🌐 Language")

    gr.Markdown("""## 🎵 Beethoven
音乐多乐器声源分离 · Music Source Separation""")

    with gr.Row():
        with gr.Column(scale=1):
            upload_box = gr.File(label=_('upload','zh'), file_types=[".mp3",".wav",".m4a",".flac",".ogg"])
            model_radio = gr.Radio(choices=list(MODELS.keys()), value=list(MODELS.keys())[0], label=_('model','zh'))
            model_desc = gr.Markdown("**Demucs 4-stem**: 人声/贝斯/鼓/其他")

            sep_btn = gr.Button(_('separate','zh'), variant="primary", size="lg")

            with gr.Group():
                orig_audio_box = gr.Audio(label=_('original','zh'), type="filepath")
                audio_boxes = {}
                for inst in ALL_INSTRUMENTS:
                    audio_boxes[inst] = gr.Audio(label=iname(inst,'zh'), type="filepath", visible=False)

            # MIDI download section (hidden by default)
            midi_info = gr.Markdown("", visible=False)

        with gr.Column(scale=1):
            spec_output = gr.Image(label=_('spec','zh'), type="filepath", height=650)
            status_box = gr.Textbox(label=_('status','zh'), interactive=False, lines=8)

    gr.Markdown(_('footer','zh'))

    # ========== Events ==========
    def on_model_change(m, lang):
        """Update UI based on selected model."""
        cfg = MODELS[m]
        insts = cfg['instruments']
        descs = {
            '4-stem (人声/贝斯/鼓/其他)': '**Demucs** · 分离为人声/贝斯/鼓/其他',
            '6-stem (+吉他/钢琴)': '**Demucs** · 分离为 6 轨 (+吉他/钢琴)',
            '构想二 (生成式 + MIDI)': '**构想二** · 先听谱再演奏，生成 MIDI 乐谱 🎵',
        }
        updates = {}
        for inst in ALL_INSTRUMENTS:
            updates[audio_boxes[inst]] = gr.Audio.update(
                label=iname(inst,lang), visible=inst in insts)
        updates[model_desc] = gr.Markdown.update(descs.get(m, ''))
        updates[midi_info] = gr.Markdown.update(visible=(cfg['mode']=='generative'))
        return updates

    def on_lang(lang, model):
        cfg = MODELS[model]
        insts = cfg['instruments']
        updates = {
            upload_box: gr.File.update(label=_('upload',lang)),
            model_radio: gr.Radio.update(label=_('model',lang)),
            sep_btn: gr.Button.update(value=_('separate',lang)),
            orig_audio_box: gr.Audio.update(label=_('original',lang)),
            spec_output: gr.Image.update(label=_('spec',lang)),
            status_box: gr.Textbox.update(label=_('status',lang)),
        }
        for inst in insts:
            updates[audio_boxes[inst]] = gr.Audio.update(label=iname(inst,lang))
        return updates

    model_radio.change(on_model_change, [model_radio, lang_radio],
                       [model_desc, midi_info] + list(audio_boxes.values()))

    lang_radio.change(on_lang, [lang_radio, model_radio],
                      [upload_box, model_radio, sep_btn, orig_audio_box,
                       spec_output, status_box] + list(audio_boxes.values()))

    sep_event = sep_btn.click(
        process, [upload_box, model_radio, lang_radio],
        [spec_output, orig_audio_box,
         audio_boxes['vocals'], audio_boxes['bass'], audio_boxes['drums'],
         audio_boxes['guitar'], audio_boxes['piano'], audio_boxes['other'],
         status_box]
    )

if __name__ == "__main__":
    import sys as _sys
    if hasattr(_sys.stdout,'reconfigure'): _sys.stdout.reconfigure(encoding='utf-8',errors='replace')
    print(f"{'='*50}")
    print(f"  🎵 Beethoven - 音乐多乐器声源分离")
    print(f"{'='*50}")
    print(f"  📡 启动 Web 界面...")
    print(f"  🌐 地址: http://127.0.0.1:7865")
    print(f"  📁 输出目录: {OUTPUT_BASE}")
    print(f"  ⚙️  设备: {DEVICE}")
    print(f"{'='*50}")
    print(f"  在浏览器中打开上面的地址即可使用")
    if not getattr(_sys, 'frozen', False):
        import webbrowser
        webbrowser.open("http://127.0.0.1:7865")
    app.launch(server_name="127.0.0.1", server_port=7865, share=False,
               theme=gr.themes.Soft(primary_hue="orange", neutral_hue="slate"))
