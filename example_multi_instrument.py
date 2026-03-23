"""
Exemplo: renderizar MIDI com múltiplos instrumentos,
cada faixa usando um banco de samples diferente.
"""
from pathlib import Path
from midi_renderer import SampleBank, MidiRenderer
import pretty_midi
import soundfile as sf
import numpy as np

# Mapeamento: número de programa MIDI → pasta de samples
INSTRUMENT_BANKS = {
    "violin":   "./samples/violin",
    "cello":    "./samples/cello",
    "flute":    "./samples/flute",
    "trumpet":  "./samples/trumpet",
    "piano":    "./samples/piano",
}

# Mapeamento de programa MIDI para instrumento
def program_to_instrument(program: int) -> str:
    if program in range(0, 8):    return "piano"
    if program in range(40, 48):  return "violin"
    if program in range(32, 40):  return "cello"
    if program in range(72, 80):  return "flute"
    if program in range(56, 64):  return "trumpet"
    return "piano"  # fallback


def render_multi_instrument(midi_path: str, output_path: str, sr: int = 44100):
    pm = pretty_midi.PrettyMIDI(midi_path)
    total_samples = int(pm.get_end_time() * sr) + sr
    mix = np.zeros(total_samples, dtype=np.float32)

    for instrument in pm.instruments:
        if instrument.is_drum:
            continue

        inst_name = program_to_instrument(instrument.program)
        bank_path = INSTRUMENT_BANKS.get(inst_name)

        if not bank_path or not Path(bank_path).exists():
            print(f"Banco não encontrado para '{inst_name}'. Pulando.")
            continue

        print(f"Renderizando '{instrument.name}' com banco '{inst_name}'...")
        bank = SampleBank(bank_path)

        # Cria um renderer sem gravar arquivo (acesso direto ao buffer)
        renderer = MidiRenderer(bank, output_sr=sr)
        for note in instrument.notes:
            renderer._render_note(mix, note, instrument.program)

    # Normaliza
    peak = np.max(np.abs(mix))
    if peak > 0:
        mix = mix / peak * 0.9

    sf.write(output_path, mix, sr)
    print(f"Arquivo gerado: {output_path}")


if __name__ == "__main__":
    render_multi_instrument("sua_musica.mid", "saida_multi.wav")
