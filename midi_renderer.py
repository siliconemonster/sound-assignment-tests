"""
MIDI Sample Renderer
====================
Renderiza arquivos MIDI usando bancos de samples de instrumentos reais.
Usa detecção automática de loop points para sustain natural sem reataque.

Dependências:
    pip install mido numpy soundfile scipy librosa pretty_midi
"""

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
    logging.warning("librosa não encontrado. Pitch-shifting de alta qualidade indisponível.")

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

DYNAMICS_ORDER = ["pppp", "ppp", "pp", "p", "mp", "mf", "f", "ff", "fff", "ffff"]

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

ARTICULATIONS = [
    "arco", "pizz", "pizzicato",
    "sul_tasto", "sul_ponticello",
    "harmonics", "flautando",
    "staccato", "normal", "long", "short",
    "vibrato", "nonvibrato", "non_vibrato",
    "flutter", "multiphonic",
    "trem", "tremolo",
    "col_legno", "mute", "muted", "open",
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
# Loop point detection
# ---------------------------------------------------------------------------

def find_loop_points(audio: np.ndarray, sr: int) -> tuple[int, int]:
    """
    Detecta automaticamente loop_start e loop_end no trecho de sustain do sample.

    Estratégia:
      1. Encontra o fim do ataque (após o pico de amplitude)
      2. No trecho de sustain, busca dois pontos com cruzamento de zero
         e correlação de forma de onda alta — garante loop sem clique
      3. Retorna (loop_start, loop_end) em amostras

    Se não encontrar pontos adequados, retorna um loop que cobre
    a maior parte do sustain disponível.
    """
    n = len(audio)
    if n < sr * 0.1:
        # Sample muito curto: loop em toda a extensão
        return 0, n - 1

    # --- 1. Fim do ataque ---
    peak_idx = int(np.argmax(np.abs(audio)))
    # Ataque termina após o pico + 20ms de margem
    attack_end = min(peak_idx + int(0.02 * sr), int(n * 0.5))

    # --- 2. Fim do sustain (onde amplitude cai abaixo de 20% do pico) ---
    peak_amp = np.max(np.abs(audio))
    threshold = peak_amp * 0.20
    above = np.where(np.abs(audio[attack_end:]) > threshold)[0]
    if len(above) == 0:
        sustain_end = attack_end + int(0.1 * sr)
    else:
        sustain_end = attack_end + int(above[-1])
    sustain_end = min(sustain_end, n - 1)

    sustain_len = sustain_end - attack_end
    if sustain_len < int(0.05 * sr):
        # Sustain muito curto: loop no que tiver
        return attack_end, sustain_end

    # --- 3. Tamanho do segmento de busca ---
    # Usa janelas de ~50ms para comparação
    win = int(0.05 * sr)
    win = max(win, 64)

    # Região de busca: segunda metade do sustain
    search_start = attack_end + sustain_len // 3
    search_end   = sustain_end - win

    if search_start >= search_end:
        return attack_end, sustain_end

    # --- 4. Busca por zero-crossings próximos com boa correlação ---
    # Encontra zero-crossings na região de busca
    region = audio[search_start:search_end]
    zc = np.where(np.diff(np.sign(region)))[0] + search_start

    if len(zc) < 2:
        return attack_end, sustain_end

    best_score = -1.0
    best_start = attack_end
    best_end   = sustain_end

    # Testa pares de zero-crossings espaçados por pelo menos 20ms
    min_gap = int(0.02 * sr)
    max_candidates = 30  # limita busca para não demorar demais

    candidates = zc[::max(1, len(zc) // max_candidates)]

    for i, ls in enumerate(candidates):
        for le in candidates[i + 1:]:
            if le - ls < min_gap:
                continue
            if le + win > n:
                break
            # Correlação entre o trecho logo antes de loop_end
            # e o trecho logo após loop_start
            seg_start = audio[ls:ls + win]
            seg_end   = audio[le:le + win]
            if np.std(seg_start) < 1e-6 or np.std(seg_end) < 1e-6:
                continue
            corr = float(np.corrcoef(seg_start, seg_end)[0, 1])
            if corr > best_score:
                best_score = corr
                best_start = ls
                best_end   = le

    log.debug(f"Loop points: start={best_start} end={best_end} corr={best_score:.3f}")
    return best_start, best_end


# ---------------------------------------------------------------------------
# Estrutura de sample
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    path: Path
    midi_note: int
    dynamic: str = "mf"
    articulation: str = "normal"
    duration_hint: Optional[float] = None

    _audio: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    _sr: Optional[int] = field(default=None, repr=False, compare=False)
    _loop: Optional[tuple[int, int]] = field(default=None, repr=False, compare=False)

    def load(self) -> tuple[np.ndarray, int]:
        if self._audio is None:
            audio, sr = sf.read(str(self.path), always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            self._audio = audio.astype(np.float32)
            self._sr = sr
        return self._audio, self._sr

    def get_loop_points(self, sr: int) -> tuple[int, int]:
        """Retorna loop points, detectando se ainda não foram calculados."""
        if self._loop is None:
            audio, _ = self.load()
            self._loop = find_loop_points(audio, sr)
        return self._loop


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
            # Ignora frases — contêm múltiplas notas, não são samples individuais
            if "phrase" in path.stem.lower():
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

        for m in re.finditer(r'(?:^|_)([A-Ga-g][#b]?)(-?\d)(?:_|$)', stem):
            candidate = note_name_to_midi(m.group(1).upper() + m.group(2))
            if candidate is not None:
                midi_note = candidate
                break

        if midi_note is None:
            for m in re.finditer(r'\b([A-Ga-g][#b]?)(-?\d)\b', stem):
                candidate = note_name_to_midi(m.group(1).upper() + m.group(2))
                if candidate is not None:
                    midi_note = candidate
                    break

        if midi_note is None:
            for m in re.finditer(r'(?<!\d)(\d{2,3})(?!\d)', stem):
                n = int(m.group(1))
                if 0 <= n <= 127:
                    midi_note = n
                    break

        if midi_note is None:
            return None

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

        articulation = "normal"
        for a in sorted(ARTICULATIONS, key=len, reverse=True):
            if a in stem_lower:
                articulation = a
                break

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
            if target_duration is not None and s.duration_hint is not None:
                if s.duration_hint >= target_duration:
                    dur_dist = s.duration_hint - target_duration
                else:
                    dur_dist = (target_duration - s.duration_hint) * 10
            elif target_duration is not None and s.duration_hint is None:
                dur_dist = 5.0
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
# Motor de áudio
# ---------------------------------------------------------------------------

class AudioProcessor:

    @staticmethod
    def pitch_shift(audio: np.ndarray, sr: int, semitones: float) -> np.ndarray:
        if abs(semitones) < 0.05:
            return audio
        if HAS_LIBROSA:
            return librosa.effects.pitch_shift(audio, sr=sr, n_steps=semitones)
        else:
            ratio = 2.0 ** (semitones / 12.0)
            new_len = max(1, int(len(audio) / ratio))
            indices = np.linspace(0, len(audio) - 1, new_len)
            return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

    @staticmethod
    def apply_loop(audio: np.ndarray, sr: int,
                   loop_start: int, loop_end: int,
                   target_duration: float) -> np.ndarray:
        """
        Monta o áudio final usando loop points:
          [0 .. loop_start]  — ataque (toca uma vez)
          [loop_start .. loop_end]  — sustain (loopa até target_duration)
          [loop_end .. fim]  — release (decaimento natural, toca uma vez)

        O crossfade na junção do loop evita cliques.
        """
        target_samples = int(target_duration * sr)
        loop_seg = audio[loop_start:loop_end]
        loop_len = len(loop_seg)

        if loop_len < 16:
            # Loop muito curto: fallback para sample completo
            return audio

        attack  = audio[:loop_start]
        release = audio[loop_end:]

        # Quantas amostras de sustain precisamos
        sustain_needed = max(0, target_samples - len(attack) - len(release))

        if sustain_needed <= 0:
            # Nota mais curta que ataque+release: usa sample direto
            total = len(attack) + len(release)
            result = np.concatenate([attack, release])
            return result

        # Monta sustain por loop com crossfade
        xfade = min(int(0.01 * sr), loop_len // 4, 256)  # ~10ms
        loops_needed = math.ceil(sustain_needed / loop_len)
        raw_sustain = np.tile(loop_seg, loops_needed + 1)

        # Aplica crossfade nas junções
        for i in range(1, loops_needed + 1):
            pos = i * loop_len
            if pos + xfade >= len(raw_sustain):
                break
            fade_out = np.linspace(1.0, 0.0, xfade) ** 2
            fade_in  = np.linspace(0.0, 1.0, xfade) ** 2
            raw_sustain[pos:pos + xfade] = (
                raw_sustain[pos:pos + xfade] * fade_out +
                loop_seg[:xfade] * fade_in
            )

        sustain = raw_sustain[:sustain_needed]

        return np.concatenate([attack, sustain, release]).astype(np.float32)

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
        total_samples = int(total_seconds * self.output_sr) + self.output_sr * 3

        mix = [np.zeros(total_samples, dtype=np.float32)]

        for instrument in pm.instruments:
            kind = "percussão" if instrument.is_drum else "instrumento"
            log.info(f"  Renderizando {kind}: '{instrument.name}' ({len(instrument.notes)} notas)")
            for note in instrument.notes:
                mix[0] = self._render_note(mix[0], note)

        mix = mix[0]

        # Apara silêncio no final — encontra última amostra acima de -80dB
        threshold = np.max(np.abs(mix)) * 0.0001
        nonsilent = np.where(np.abs(mix) > threshold)[0]
        if len(nonsilent) > 0:
            # Mantém 300ms de cauda para o release da última nota decair
            tail = min(int(0.3 * self.output_sr), len(mix) - nonsilent[-1])
            mix = mix[:nonsilent[-1] + tail]

        # Normaliza DEPOIS de aparar, com fade-out nos últimos 50ms
        # para garantir que a última nota não clipe
        fade_len = min(int(0.05 * self.output_sr), len(mix))
        mix[-fade_len:] *= np.linspace(1.0, 0.0, fade_len) ** 2

        peak = np.max(np.abs(mix))
        if peak > 0:
            mix = mix / peak * 0.85

        log.info(f"Exportando: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), mix, self.output_sr)
        log.info("Concluído!")

    def _render_note(self, mix: np.ndarray, note) -> np.ndarray:
        target_note = note.pitch
        duration    = note.end - note.start
        velocity    = note.velocity
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

        # Resample para output_sr
        if sr != self.output_sr:
            if HAS_LIBROSA:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.output_sr)
            else:
                ratio = self.output_sr / sr
                new_len = max(1, int(len(audio) * ratio))
                indices = np.linspace(0, len(audio) - 1, new_len)
                audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
            # Recalcula loop points com o novo sr
            loop_start, loop_end = find_loop_points(audio, self.output_sr)
        else:
            loop_start, loop_end = sample.get_loop_points(self.output_sr)

        # Pitch shift
        semitones = target_note - sample.midi_note
        if abs(semitones) >= 0.05:
            audio = AudioProcessor.pitch_shift(audio, self.output_sr, semitones)
            # Loop points precisam ser reescalados se o pitch shift mudou o tamanho
            ratio = len(audio) / max(1, len(sample._audio))
            loop_start = int(loop_start * ratio)
            loop_end   = int(loop_end   * ratio)
            loop_start = max(0, min(loop_start, len(audio) - 2))
            loop_end   = max(loop_start + 1, min(loop_end, len(audio) - 1))

        # Aplica loop points para montar o áudio na duração certa
        audio = AudioProcessor.apply_loop(
            audio, self.output_sr, loop_start, loop_end, duration
        )

        # Velocity
        audio = AudioProcessor.apply_velocity_envelope(audio, velocity)

        # Mix
        end_sample = start_sample + len(audio)
        if end_sample > len(mix):
            extra = np.zeros(end_sample - len(mix), dtype=np.float32)
            mix = np.concatenate([mix, extra])
        mix[start_sample:end_sample] += audio

        # Soft clip em tempo real para evitar distorção por sobreposição de notas
        np.clip(mix, -1.0, 1.0, out=mix)
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
