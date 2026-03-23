# Sound Assignment Tests

## MIDI Sample Renderer

Renderiza arquivos MIDI usando bancos de samples de instrumentos reais (Philharmonia, IRCAM FullSOL ou qualquer coleção organizada por nota).

---

### Instalação

```bash
pip install mido numpy soundfile scipy librosa pretty_midi
```

> `librosa` é recomendado para pitch-shifting e time-stretching de alta qualidade.
> O programa funciona sem ele, mas com qualidade inferior.

---

### Uso via linha de comando

```bash
python midi_renderer.py <arquivo.mid> <pasta_de_samples> [opções]
```

### Exemplo básico

```bash
python midi_renderer.py teste.mid samples\Philharmonia\trombone -o saida.wav
```

### Outros exemplos

```bash
# Especificar articulação (arco, pizz, staccato, etc.)
python midi_renderer.py teste.mid samples\Philharmonia\violin -o saida.wav --articulation arco

# Saída em 48kHz
python midi_renderer.py teste.mid samples\Philharmonia\violin -o saida.wav --sr 48000

# Listar todos os samples detectados no banco
python midi_renderer.py teste.mid samples\Philharmonia\violin --list-samples
```

---

### Uso como biblioteca Python

```python
from midi_renderer import SampleBank, MidiRenderer

bank = SampleBank("./samples/Philharmonia/violin")
renderer = MidiRenderer(bank, output_sr=44100)
renderer.render("teste.mid", "saida.wav")
```

---

### Formatos de nome de arquivo suportados

#### Philharmonia Orchestra
```
trombone_A2_1_mezzo-forte_normal.mp3
violin_Bb4_05_pianissimo_arco-normal.mp3
flute_C5_very-long_forte_normal.mp3
```

O número no nome indica a duração do sample (`_025_` = 0.25s, `_05_` = 0.5s, `_1_` = 1s, `_15_` = 1.5s, `very-long` ≈ 8s). O programa usa essa informação para escolher o sample de duração mais adequada para cada nota MIDI.

#### IRCAM FullSOL
```
Violin-A4-mf.wav
Cello-F#3-ff.wav
Trumpet-G4-pp.wav
```

#### Genérico
Qualquer arquivo que contenha nota + oitava no nome:
```
piano_C4_medium.wav
guitar_note_A3.flac
brass_F5_loud.aif
```

---

### Parâmetros CLI

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `midi` | obrigatório | Arquivo MIDI de entrada |
| `samples_dir` | obrigatório | Diretório raiz do banco de samples |
| `-o / --output` | `<midi>.wav` | Arquivo WAV de saída |
| `--sr` | 44100 | Taxa de amostragem de saída (Hz) |
| `--max-dist` | 12 | Máximo de semitons para busca de sample mais próximo |
| `--articulation` | normal | Articulação preferida (arco, pizz, staccato…) |
| `--list-samples` | — | Lista samples detectados e encerra |

---

### Como funciona

1. **Escaneamento** — O banco é escaneado recursivamente; nota, dinâmica, articulação e duração são extraídas do nome de cada arquivo.
2. **Busca** — Para cada nota MIDI, o sample mais próximo é selecionado por: articulação → dinâmica → distância semitonal → duração compatível.
3. **Pitch-shift** — Se o sample não está na nota exata, `librosa.effects.pitch_shift` o transpõe (em semitons) sem alterar a duração.
4. **Duração** — Se a nota MIDI for mais longa que o sample, aplica time-stretch. Se for mais curta, o sample toca até o fim natural.
5. **Velocity** — O volume de cada nota é mapeado do velocity MIDI (0–127) para um ganho de amplitude com curva suave.
6. **Mix** — Todas as notas são somadas em um buffer e exportadas como WAV normalizado.

---

### Estrutura sugerida de pastas para múltiplos instrumentos

```
samples/
└── Philharmonia/
    ├── violin/
    ├── cello/
    ├── flute/
    ├── trombone/
    └── ...
```

Para um MIDI com múltiplos instrumentos, renderize faixa por faixa e mixe depois,
ou modifique `MidiRenderer` para mapear `program_number → SampleBank`.