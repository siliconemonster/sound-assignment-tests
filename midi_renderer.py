"""
MIDI Sample Renderer
====================
Renderiza arquivos MIDI usando bancos de samples de instrumentos reais.
Compatível com os formatos da Philharmonia Orchestra e IRCAM FullSOL.

Dependências:
    pip install mido numpy soundfile scipy librosa pretty_midi
"""

import os
import re
import math
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import soundfile as sf
import mido
import pretty_midi

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    logging.warning("librosa não encontrado. Time-stretching de alta qualidade indisponível.")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes MIDI
# ---------------------------------------------------------------------------
MIDI_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

NOTE_NAME_TO_SEMITONE = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}

# Dinâmicas reconhecidas nos nomes de arquivo
DYNAMICS_ORDER = ["pppp", "ppp", "pp", "p", "mp", "mf", "f", "ff", "fff", "ffff"]

# Mapeamento dos nomes por extenso usados pela Philharmonia → abreviação padrão
PHILHARMONIA_DYNAMICS = {
    "pianississimo": "ppp",
    "pianissimo":    "pp",
    "mezzo-piano":   "mp",
    "mezzo_piano":   "mp",
    "mezzopiano":    "mp",
    "mezzo-forte":   "mf",
    "mezzo_forte":   "mf",
    "mezzoforte":    "mf",
    "fortissimo":    "ff",
    "fortississimo": "fff",
    "forte":         "f",
    "piano":         "p",
}

# Articulações reconhecidas
ARTICULATIONS = [
    "arco", "pizz", "pizzicato",
    "sul_tasto", "sul_ponticello",
    "harmonics", "flautando",
    "staccato", "normal", "long", "short",
    "vibrato", "nonvibrato", "non_vibrato",
    "flutter", "multiphonic",
    "trem", "tremolo",
    "col_legno",
    "mute", "muted",
    "open",
]


# ---------------------------------------------------------------------------
# Helpers de nota
# ---------------------------------------------------------------------------

def midi_note_to_name(midi_note: int) -> str:
    octave = (midi_note // 12) - 1
    note = MIDI_NOTE_NAMES[midi_note % 12]
    return f"{note}{octave}"


def note_name_to_midi(note_name: str) -> Optional[int]:
    note_name = note_name.replace("As", "A#").replace("Es", "Eb").replace("Bes", "Bb")
    m = re.match(r'^([A-Ga-g][#b]?)(-?\d+)$', note_name)
    if not m:
        return None
    pitch, octave_str = m.group(1), m.group(2)
    pitch = pitch[0].upper() + pitch[1:]
    if pitch not in NOTE_NAME_TO_SEMITONE:
        return None
    return (int(octave_str) + 1) * 12 + NOTE_NAME_TO_SEMITONE[pitch]


# ---------------------------------------------------------------------------
# Estrutura de sample
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    path: Path
    midi_note: int
    dynamic: str = "mf"
    articulation: str = "normal"
    duration_hint: Optional[float] = None  # duração em segundos indicada no nome do arquivo

    _audio: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    _sr: Optional[int] = field(default=None, repr=False, compare=False)

    def load(self) -> tuple[np.ndarray, int]:
        if self._audio is None:
            audio, sr = sf.read(str(self.path), always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            self._audio = audio.astype(np.float32)
            self._sr = sr
        return self._audio, self._sr


# ---------------------------------------------------------------------------
# Banco de samples
# ---------------------------------------------------------------------------

class SampleBank:
    AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".aif", ".aiff", ".ogg"}

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.samples: list[Sample] = []
        self._index: dict[int, list[Sample]] = {}
        self._scan()

    def _scan(self):
        log.info(f"Escaneando banco de samples em: {self.root}")
        count = 0
        for path in sorted(self.root.rglob("*")):
            if path.suffix.lower() not in self.AUDIO_EXTENSIONS:
                continue
            sample = self._parse_filename(path)
            if sample:
                self.samples.append(sample)
                self._index.setdefault(sample.midi_note, []).append(sample)
                count += 1
        log.info(f"  {count} samples carregados, {len(self._index)} notas distintas.")

    def _parse_filename(self, path: Path) -> Optional[Sample]:
        stem = path.stem
        stem_lower = stem.lower()

        midi_note = None

        # Padrão 1: nota+oitava entre underscores (Philharmonia: _A2_, _Bb4_)
        for m in re.finditer(r'(?:^|_)([A-Ga-g][#b]?)(-?\d)(?:_|$)', stem):
            candidate = note_name_to_midi(m.group(1).upper() + m.group(2))
            if candidate is not None:
                midi_note = candidate
                break

        # Padrão 2: nota+oitava como palavra isolada
        if midi_note is None:
            for m in re.finditer(r'\b([A-Ga-g][#b]?)(-?\d)\b', stem):
                candidate = note_name_to_midi(m.group(1).upper() + m.group(2))
                if candidate is not None:
                    midi_note = candidate
                    break

        # Padrão 3: número MIDI puro
        if midi_note is None:
            for m in re.finditer(r'(?<!\d)(\d{2,3})(?!\d)', stem):
                n = int(m.group(1))
                if 0 <= n <= 127:
                    midi_note = n
                    break

        if midi_note is None:
            return None

        # Dinâmica
        dynamic = "mf"
        matched_dyn = False
        for name, abbr in PHILHARMONIA_DYNAMICS.items():
            if name in stem_lower:
                dynamic = abbr
                matched_dyn = True
                break
        if not matched_dyn:
            for d in sorted(DYNAMICS_ORDER, key=len, reverse=True):
                if re.search(r'(?<![a-z])' + d + r'(?![a-z])', stem_lower):
                    dynamic = d
                    break

        # Articulação
        articulation = "normal"
        for a in sorted(ARTICULATIONS, key=len, reverse=True):
            if a in stem_lower:
                articulation = a
                break

        # Duração do sample indicada no nome do arquivo (Philharmonia)
        duration_hint = None
        dur_map = {
            'very-long': 8.0, 'very_long': 8.0,
            'long': 3.0,
            '_15_': 1.5, '-15-': 1.5,
            '_1_': 1.0,  '-1-': 1.0,
            '_05_': 0.5, '-05-': 0.5,
            '_025_': 0.25, '-025-': 0.25,
        }
        for key, dur in dur_map.items():
            if key in stem_lower:
                duration_hint = dur
                break

        return Sample(path=path, midi_note=midi_note, dynamic=dynamic,
                      articulation=articulation, duration_hint=duration_hint)

    def find_closest(
        self,
        target_note: int,
        dynamic: str = "mf",
        articulation: str = "normal",
        max_semitone_distance: int = 12,
        target_duration: float = None,
    ) -> Optional[Sample]:
        if not self.samples:
            return None

        candidates = self.samples

        filtered = [s for s in candidates if s.articulation == articulation]
        if filtered:
            candidates = filtered

        def score(s: Sample) -> tuple:
            dist = abs(s.midi_note - target_note)
            dyn_dist = abs(
                DYNAMICS_ORDER.index(s.dynamic) - DYNAMICS_ORDER.index(dynamic)
                if s.dynamic in DYNAMICS_ORDER and dynamic in DYNAMICS_ORDER else 99
            )
            # Prefere sample com duração >= duração da nota MIDI
            # mas sem ser excessivamente longo
            if target_duration is not None and s.duration_hint is not None:
                if s.duration_hint >= target_duration:
                    dur_dist = s.duration_hint - target_duration  # menor é melhor
                else:
                    dur_dist = (target_duration - s.duration_hint) * 10  # penaliza curtos
            elif target_duration is not None and s.duration_hint is None:
                dur_dist = 5.0  # sample sem info de duração: penalidade média
            else:
                dur_dist = 0
            return (dist, dyn_dist, dur_dist)

        candidates.sort(key=score)
        best = candidates[0]
        dist = abs(best.midi_note - target_note)

        if dist > max_semitone_distance:
            log.warning(
                f"Nota {target_note} ({midi_note_to_name(target_note)}): "
                f"sample mais próximo está a {dist} semitons ({best.path.name})"
            )

        return best


# ---------------------------------------------------------------------------
# Motor de pitch-shift / time-stretch
# ---------------------------------------------------------------------------

class AudioProcessor:

    @staticmethod
    def semitones_to_ratio(semitones: float) -> float:
        return 2.0 ** (semitones / 12.0)

    @staticmethod
    def pitch_shift(audio: np.ndarray, sr: int, semitones: float) -> np.ndarray:
        if abs(semitones) < 0.05:
            return audio
        if HAS_LIBROSA:
            return librosa.effects.pitch_shift(audio, sr=sr, n_steps=semitones)
        else:
            ratio = AudioProcessor.semitones_to_ratio(semitones)
            new_len = max(1, int(len(audio) / ratio))
            indices = np.linspace(0, len(audio) - 1, new_len)
            return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

    @staticmethod
    def time_stretch(audio: np.ndarray, sr: int, target_duration: float) -> np.ndarray:
        current_duration = len(audio) / sr
        if current_duration <= 0:
            return audio

        ratio = current_duration / target_duration

        if abs(ratio - 1.0) < 0.02:
            return AudioProcessor._trim_or_pad(audio, int(target_duration * sr))

        if HAS_LIBROSA:
            try:
                stretched = librosa.effects.time_stretch(audio, rate=ratio)
            except Exception:
                stretched = audio
        else:
            stretched = audio

        target_samples = int(target_duration * sr)
        return AudioProcessor._trim_or_pad(stretched, target_samples)

    @staticmethod
    def _trim_or_pad(audio: np.ndarray, target_samples: int) -> np.ndarray:
        if len(audio) >= target_samples:
            result = audio[:target_samples].copy()
            fade_len = min(target_samples, int(0.01 * 44100))
            if fade_len > 0:
                result[-fade_len:] *= np.linspace(1, 0, fade_len)
            return result
        else:
            loops = math.ceil(target_samples / len(audio))
            looped = np.tile(audio, loops)[:target_samples]
            fade_len = min(target_samples, int(0.01 * 44100))
            if fade_len > 0:
                looped[-fade_len:] *= np.linspace(1, 0, fade_len)
            return looped

    @staticmethod
    def apply_velocity_envelope(audio: np.ndarray, velocity: int) -> np.ndarray:
        gain = (velocity / 127.0) ** 1.5
        return (audio * gain).astype(np.float32)


# ---------------------------------------------------------------------------
# Renderizador principal
# ---------------------------------------------------------------------------

class MidiRenderer:

    def __init__(
        self,
        bank: SampleBank,
        output_sr: int = 44100,
        max_semitone_distance: int = 12,
        default_articulation: str = "normal",
    ):
        self.bank = bank
        self.output_sr = output_sr
        self.max_semitone_distance = max_semitone_distance
        self.default_articulation = default_articulation

    def render(self, midi_path: str | Path, output_path: str | Path):
        midi_path = Path(midi_path)
        output_path = Path(output_path)

        log.info(f"Carregando MIDI: {midi_path}")
        pm = pretty_midi.PrettyMIDI(str(midi_path))
        total_seconds = pm.get_end_time()
        total_samples = int(total_seconds * self.output_sr) + self.output_sr

        mix = [np.zeros(total_samples, dtype=np.float32)]

        for instrument in pm.instruments:
            if instrument.is_drum:
                continue
            log.info(f"  Renderizando: '{instrument.name}' ({len(instrument.notes)} notas)")
            for note in instrument.notes:
                mix[0] = self._render_note(mix[0], note, instrument.program)

        mix = mix[0]
        peak = np.max(np.abs(mix))
        if peak > 0:
            mix = mix / peak * 0.9

        log.info(f"Exportando: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), mix, self.output_sr)
        log.info("Concluído!")

    def _render_note(self, mix: np.ndarray, note, program: int):
        target_note = note.pitch
        duration = note.end - note.start
        velocity = note.velocity
        start_sample = int(note.start * self.output_sr)

        dynamic = self._velocity_to_dynamic(velocity)

        sample = self.bank.find_closest(
            target_note,
            dynamic=dynamic,
            articulation=self.default_articulation,
            max_semitone_distance=self.max_semitone_distance,
            target_duration=duration,
        )
        if sample is None:
            log.warning(f"Nenhum sample para nota {target_note}. Pulando.")
            return mix

        try:
            audio, sr = sample.load()
        except Exception as e:
            log.error(f"Erro ao carregar {sample.path}: {e}")
            return mix

        # Resample
        if sr != self.output_sr:
            if HAS_LIBROSA:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.output_sr)
            else:
                ratio = self.output_sr / sr
                new_len = max(1, int(len(audio) * ratio))
                indices = np.linspace(0, len(audio) - 1, new_len)
                audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

        # Pitch shift
        semitones = target_note - sample.midi_note
        audio = AudioProcessor.pitch_shift(audio, self.output_sr, semitones)

        # Duração: o sample toca sempre até o fim natural.
        # Se a nota MIDI for mais longa que o sample, estende.
        # Se for mais curta, não corta — o decaimento do instrumento faz o legato.
        sample_duration = len(audio) / self.output_sr
        if duration > sample_duration:
            audio = AudioProcessor.time_stretch(audio, self.output_sr, duration)
        # else: usa o sample completo sem cortar

        # Velocity
        audio = AudioProcessor.apply_velocity_envelope(audio, velocity)

        # Mix — expande o buffer se o sample for além do fim previsto do MIDI
        end_sample = start_sample + len(audio)
        if end_sample > len(mix):
            extra = np.zeros(end_sample - len(mix), dtype=np.float32)
            mix = np.concatenate([mix, extra])
        mix[start_sample:end_sample] += audio
        return mix

    @staticmethod
    def _velocity_to_dynamic(velocity: int) -> str:
        if velocity < 16:   return "pppp"
        if velocity < 33:   return "ppp"
        if velocity < 49:   return "pp"
        if velocity < 64:   return "p"
        if velocity < 80:   return "mp"
        if velocity < 96:   return "mf"
        if velocity < 112:  return "f"
        if velocity < 120:  return "ff"
        return "fff"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Renderiza arquivo MIDI usando banco de samples de instrumentos."
    )
    parser.add_argument("midi", help="Caminho para o arquivo MIDI (.mid/.midi)")
    parser.add_argument("samples_dir", help="Diretório raiz do banco de samples")
    parser.add_argument("-o", "--output", default=None, help="Arquivo de saída WAV")
    parser.add_argument("--sr", type=int, default=44100, help="Taxa de amostragem (padrão: 44100)")
    parser.add_argument("--max-dist", type=int, default=12, help="Máximo de semitons de distância")
    parser.add_argument("--articulation", default="normal", choices=ARTICULATIONS)
    parser.add_argument("--list-samples", action="store_true", help="Lista samples e sai")
    args = parser.parse_args()

    bank = SampleBank(args.samples_dir)

    if args.list_samples:
        print(f"\n{'Nota':>5}  {'Dinâmica':>6}  {'Articulação':>15}  Arquivo")
        print("-" * 80)
        for s in sorted(bank.samples, key=lambda x: x.midi_note):
            print(f"{s.midi_note:>5}  {s.dynamic:>6}  {s.articulation:>15}  {s.path.name}")
        return

    output = args.output or Path(args.midi).with_suffix(".wav")

    renderer = MidiRenderer(
        bank=bank,
        output_sr=args.sr,
        max_semitone_distance=args.max_dist,
        default_articulation=args.articulation,
    )
    renderer.render(args.midi, output)


if __name__ == "__main__":
    main()