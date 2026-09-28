import os
import requests
import re
import threading
import time
import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk, filedialog
from bs4 import BeautifulSoup
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
from datetime import datetime
import gc
import json

# --- 1. НАСТРОЙКИ ---
class Config:
    def __init__(self):
        self.batch_size = 12
        self.block_size = 128
        self.max_iters = 10000
        self.learning_rate = 5e-4
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.n_embd = 256
        self.n_head = 8
        self.n_layer = 6
        self.eval_interval = 100
        self.eval_iters = 50
        self.grad_accum_steps = 1       # Gradient accumulation
        self.checkpoint_every = 500     # Сохранять чекпоинт каждые N шагов
        self.dropout = 0.1

config = Config()
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Токен разделения документов (сброс внимания)
S_TEXT_TOKEN = "<s_text>"

# --- 2. АРХИТЕКТУРА ---
class Head(nn.Module):
    def __init__(self, head_size, dropout=0.1):
        super().__init__()
        self.key   = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, dropout=0.1):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, dropout) for _ in range(num_heads)])
        self.proj  = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.1):
        super().__init__()
        head_size = n_embd // n_head
        self.sa   = MultiHeadAttention(n_head, head_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class TankGPT(nn.Module):
    def __init__(self, vocab_size, dropout=0.1):
        super().__init__()
        self.token_embedding_table    = nn.Embedding(vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config.n_embd, config.n_head, dropout) for _ in range(config.n_layer)])
        self.ln_f   = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=config.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)
        return logits, loss

    def generate_stream(self, idx, max_new_tokens, temperature=0.8, top_k=0, top_p=1.0,
                        repetition_penalty=1.0, eos_token_ids=None):
        """
        Генератор: отдаёт по одному токену (int) за раз через yield.
        Поддерживает: temperature, top_k, top_p, repetition_penalty, eos_token_ids.
        temperature=0.0 → greedy (argmax).
        """
        if eos_token_ids is None:
            eos_token_ids = []

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # (B, vocab_size)

            # Repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(idx[0].tolist()):
                    if logits[0, token_id] < 0:
                        logits[0, token_id] *= repetition_penalty
                    else:
                        logits[0, token_id] /= repetition_penalty

            # Temperature / greedy
            if temperature == 0.0:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature

                # Top-K
                if top_k > 0:
                    top_k_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < top_k_vals[:, -1:]] = float('-inf')

                # Top-P (nucleus)
                if top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
                    sorted_logits[sorted_indices_to_remove] = float('-inf')
                    logits = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)

                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)
            token_id = idx_next.item()

            yield token_id  # ← отдаём один токен

            # EOS остановка
            if token_id in eos_token_ids:
                break


# --- 3. ИНТЕРФЕЙС ---
class UltimateStation:
    def __init__(self, root):
        self.root = root
        self.root.title("⚡ P-2 ULTIMATE STATION ⚡")
        self.root.geometry("900x800")
        self.root.configure(bg="#0d0d0d")

        self.enc = tiktoken.get_encoding("cl100k_base")
        self.model    = None
        self.training = False
        self.start_time = None

        # История loss для графика
        self.loss_history = []

        self._build_ui()
        self.print_log(f"Рабочая папка: {SCRIPT_DIR}")
        self.print_log(f"Устройство: {config.device.upper()}")
        self.update_data_info()

    # =========================================================
    # UI
    # =========================================================
    def _build_ui(self):
        # Лог (общий, всегда виден)
        self.log = scrolledtext.ScrolledText(
            self.root, height=10, bg="#0a0a0a", fg="#00ff00",
            font=("Consolas", 10), insertbackground="#00ff00"
        )
        self.log.pack(padx=10, pady=(8, 2), fill=tk.BOTH, expand=True)

        # Прогресс
        self.progress = ttk.Progressbar(self.root, orient='horizontal', mode='determinate')
        self.progress.pack(padx=10, pady=2, fill=tk.X)

        # Статус
        self.status_var = tk.StringVar(value="🟢 Готов")
        tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN,
                 bg="#1a1a1a", fg="#aaaaaa").pack(side=tk.BOTTOM, fill=tk.X)

        # Вкладки
        nb = ttk.Notebook(self.root)
        nb.pack(padx=10, pady=5, fill=tk.X)

        tab_data   = tk.Frame(nb, bg="#111111")
        tab_train  = tk.Frame(nb, bg="#111111")
        tab_gen    = tk.Frame(nb, bg="#111111")
        tab_sft    = tk.Frame(nb, bg="#111111")
        tab_config = tk.Frame(nb, bg="#111111")

        nb.add(tab_data,   text=" 📦 Данные  ")
        nb.add(tab_train,  text=" 🔥 Обучение ")
        nb.add(tab_gen,    text=" 🤖 Генерация ")
        nb.add(tab_sft,    text=" 💬 SFT ")
        nb.add(tab_config, text=" ⚙️ Конфиг ")

        self._build_data_tab(tab_data)
        self._build_train_tab(tab_train)
        self._build_gen_tab(tab_gen)
        self._build_sft_tab(tab_sft)
        self._build_config_tab(tab_config)

    # ----- Вкладка: Данные -----
    def _build_data_tab(self, f):
        btn_f = tk.Frame(f, bg="#111111")
        btn_f.pack(fill=tk.X, padx=8, pady=8)

        self._btn(btn_f, "🌐 Скачать Wiki", self.fetch, "#2c3e50").pack(side=tk.LEFT, padx=4)
        self._btn(btn_f, "🔨 Склеить данные", self.tokenize, "#2980b9").pack(side=tk.LEFT, padx=4)

        # <s_text> разделитель
        sep_f = tk.Frame(f, bg="#111111")
        sep_f.pack(fill=tk.X, padx=8, pady=2)
        self.use_s_text = tk.BooleanVar(value=True)
        tk.Checkbutton(sep_f, text=f'Использовать разделитель "{S_TEXT_TOKEN}" между документами (сброс внимания)',
                       variable=self.use_s_text, bg="#111111", fg="#cccccc",
                       selectcolor="#222222", activebackground="#111111").pack(side=tk.LEFT)

        self.data_info = tk.Label(f, text="Нет данных", fg="#7f8c8d", bg="#111111")
        self.data_info.pack(anchor=tk.W, padx=10, pady=5)

    # ----- Вкладка: Обучение -----
    def _build_train_tab(self, f):
        btn_f = tk.Frame(f, bg="#111111")
        btn_f.pack(fill=tk.X, padx=8, pady=8)

        self.train_btn = self._btn(btn_f, "▶ Начать обучение", self.train, "#27ae60")
        self.train_btn.pack(side=tk.LEFT, padx=4)

        self.stop_btn = self._btn(btn_f, "⏹ Остановить", self.stop_training, "#c0392b")
        self.stop_btn.config(state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=4)

        # --- Режим: итерации или эпохи ---
        mode_f = tk.Frame(f, bg="#111111")
        mode_f.pack(fill=tk.X, padx=8, pady=4)

        self.train_mode = tk.StringVar(value="iters")

        tk.Radiobutton(mode_f, text="Итерации:", variable=self.train_mode, value="iters",
                       bg="#111111", fg="#cccccc", selectcolor="#222222",
                       activebackground="#111111", command=self._on_mode_change).pack(side=tk.LEFT, padx=4)
        self.iters_spin_var = tk.IntVar(value=10000)
        self.iters_spin = tk.Spinbox(mode_f, from_=100, to=1000000, increment=100,
                                     textvariable=self.iters_spin_var, width=8,
                                     bg="#1a1a1a", fg="#00ff00")
        self.iters_spin.pack(side=tk.LEFT, padx=2)

        tk.Radiobutton(mode_f, text="Эпох:", variable=self.train_mode, value="epochs",
                       bg="#111111", fg="#cccccc", selectcolor="#222222",
                       activebackground="#111111", command=self._on_mode_change).pack(side=tk.LEFT, padx=8)
        self.epochs_spin_var = tk.IntVar(value=10)
        self.epochs_spin = tk.Spinbox(mode_f, from_=1, to=10000, increment=1,
                                      textvariable=self.epochs_spin_var, width=6,
                                      bg="#1a1a1a", fg="#00ff00")
        self.epochs_spin.pack(side=tk.LEFT, padx=2)

        self.epoch_info_var = tk.StringVar(value="")
        tk.Label(mode_f, textvariable=self.epoch_info_var,
                 bg="#111111", fg="#f39c12", font=("Consolas", 8)).pack(side=tk.LEFT, padx=8)

        # Кнопки быстрого выбора эпох
        ep_btn_f = tk.Frame(f, bg="#111111")
        ep_btn_f.pack(fill=tk.X, padx=8, pady=2)
        tk.Label(ep_btn_f, text="Быстро:", bg="#111111", fg="#555555",
                 font=("Consolas", 8)).pack(side=tk.LEFT, padx=4)
        for ep in [1, 3, 5, 10, 20, 50]:
            tk.Button(ep_btn_f, text=f"{ep}ep",
                      command=lambda e=ep: self._set_epochs(e),
                      bg="#1a3a1a", fg="#00ff00", font=("Consolas", 8),
                      relief=tk.FLAT, padx=4, pady=1).pack(side=tk.LEFT, padx=2)

        # --- Оптимизации ---
        opt_f = tk.LabelFrame(f, text=" ⚡ Оптимизации ", bg="#111111", fg="#f39c12",
                               font=("Consolas", 8))
        opt_f.pack(fill=tk.X, padx=8, pady=4)

        self.use_amp = tk.BooleanVar(value=torch.cuda.is_available())
        tk.Checkbutton(opt_f, text="Mixed Precision FP16 (быстрее на CUDA, ~1.5-2x)",
                       variable=self.use_amp, bg="#111111", fg="#cccccc",
                       selectcolor="#222222", activebackground="#111111").pack(anchor=tk.W, padx=4)

        self.use_compile = tk.BooleanVar(value=False)
        tk.Checkbutton(opt_f, text="torch.compile() (~20-30% быстрее, PyTorch 2.0+)",
                       variable=self.use_compile, bg="#111111", fg="#cccccc",
                       selectcolor="#222222", activebackground="#111111").pack(anchor=tk.W, padx=4)

        self.use_scheduler = tk.BooleanVar(value=True)
        tk.Checkbutton(opt_f, text="Cosine LR Scheduler с warmup",
                       variable=self.use_scheduler, bg="#111111", fg="#cccccc",
                       selectcolor="#222222", activebackground="#111111").pack(anchor=tk.W, padx=4)

        # Gradient accumulation
        ga_f = tk.Frame(f, bg="#111111")
        ga_f.pack(fill=tk.X, padx=8, pady=3)
        tk.Label(ga_f, text="Gradient Accum Steps:", bg="#111111", fg="#cccccc").pack(side=tk.LEFT, padx=4)
        self.grad_accum_var = tk.IntVar(value=config.grad_accum_steps)
        tk.Spinbox(ga_f, from_=1, to=64, textvariable=self.grad_accum_var, width=5,
                   bg="#1a1a1a", fg="#00ff00").pack(side=tk.LEFT)
        tk.Label(ga_f, text="(эмулирует большой батч при малом VRAM)",
                 bg="#111111", fg="#555555", font=("Consolas", 8)).pack(side=tk.LEFT, padx=8)

        # Checkpoint каждые N эпох
        ck_f = tk.Frame(f, bg="#111111")
        ck_f.pack(fill=tk.X, padx=8, pady=3)
        tk.Label(ck_f, text="Checkpoint каждые N эпох:", bg="#111111", fg="#cccccc").pack(side=tk.LEFT, padx=4)
        self.checkpoint_var = tk.IntVar(value=1)
        tk.Spinbox(ck_f, from_=1, to=1000, increment=1, textvariable=self.checkpoint_var, width=6,
                   bg="#1a1a1a", fg="#00ff00").pack(side=tk.LEFT)

        # Loss график
        lbl_f = tk.Frame(f, bg="#111111")
        lbl_f.pack(fill=tk.X, padx=8, pady=(6, 0))
        tk.Label(lbl_f, text="📈 Loss:", bg="#111111", fg="#888888").pack(side=tk.LEFT)
        self.epoch_label_var = tk.StringVar(value="")
        tk.Label(lbl_f, textvariable=self.epoch_label_var,
                 bg="#111111", fg="#f39c12", font=("Consolas", 9)).pack(side=tk.LEFT, padx=8)

        self.loss_canvas = tk.Canvas(f, height=70, bg="#0a0a0a", highlightthickness=1,
                                     highlightbackground="#333333")
        self.loss_canvas.pack(fill=tk.X, padx=8, pady=2)

    # ----- Вкладка: Генерация -----
    def _build_gen_tab(self, f):
        # Промпт
        p_f = tk.Frame(f, bg="#111111")
        p_f.pack(fill=tk.X, padx=8, pady=8)
        self.prompt_entry = tk.Entry(p_f, width=50, font=("Consolas", 11),
                                     bg="#1a1a1a", fg="#00ff00", insertbackground="#00ff00")
        self.prompt_entry.pack(side=tk.LEFT, padx=4, expand=True, fill=tk.X)
        self.prompt_entry.insert(0, "T-34")
        self._btn(p_f, "❓ Спросить", self.test_gen, "#2980b9").pack(side=tk.RIGHT, padx=4)

        # Температура — текстовый ввод
        t_f = tk.Frame(f, bg="#111111")
        t_f.pack(fill=tk.X, padx=8, pady=3)
        tk.Label(t_f, text="🌡️ Температура (0.0=greedy, 0.1–10.0):", bg="#111111", fg="#cccccc").pack(side=tk.LEFT, padx=4)
        self.temp_var = tk.StringVar(value="0.8")
        temp_entry = tk.Entry(t_f, textvariable=self.temp_var, width=8,
                              bg="#1a1a1a", fg="#00ff00", font=("Consolas", 11),
                              insertbackground="#00ff00")
        temp_entry.pack(side=tk.LEFT, padx=4)

        # Токены на ответ — ползунок
        tok_f = tk.Frame(f, bg="#111111")
        tok_f.pack(fill=tk.X, padx=8, pady=3)
        tk.Label(tok_f, text="📝 Токенов на ответ:", bg="#111111", fg="#cccccc").pack(side=tk.LEFT, padx=4)
        self.tokens_var = tk.IntVar(value=300)
        tokens_scale = tk.Scale(tok_f, from_=10, to=2000, resolution=10,
                                orient=tk.HORIZONTAL, variable=self.tokens_var,
                                length=250, showvalue=True, bg="#111111", fg="#00ff00",
                                troughcolor="#222222", highlightthickness=0)
        tokens_scale.pack(side=tk.LEFT, padx=4)

        # Top-K
        k_f = tk.Frame(f, bg="#111111")
        k_f.pack(fill=tk.X, padx=8, pady=3)
        tk.Label(k_f, text="🔝 Top-K (0=выкл):", bg="#111111", fg="#cccccc").pack(side=tk.LEFT, padx=4)
        self.topk_var = tk.IntVar(value=0)
        tk.Scale(k_f, from_=0, to=200, resolution=1, orient=tk.HORIZONTAL,
                 variable=self.topk_var, length=200, showvalue=True,
                 bg="#111111", fg="#00ff00", troughcolor="#222222",
                 highlightthickness=0).pack(side=tk.LEFT, padx=4)

        # Top-P
        p2_f = tk.Frame(f, bg="#111111")
        p2_f.pack(fill=tk.X, padx=8, pady=3)
        tk.Label(p2_f, text="🎯 Top-P / Nucleus (1.0=выкл):", bg="#111111", fg="#cccccc").pack(side=tk.LEFT, padx=4)
        self.topp_var = tk.DoubleVar(value=1.0)
        tk.Scale(p2_f, from_=0.0, to=1.0, resolution=0.05, orient=tk.HORIZONTAL,
                 variable=self.topp_var, length=200, showvalue=True,
                 bg="#111111", fg="#00ff00", troughcolor="#222222",
                 highlightthickness=0).pack(side=tk.LEFT, padx=4)

        # Repetition Penalty
        rp_f = tk.Frame(f, bg="#111111")
        rp_f.pack(fill=tk.X, padx=8, pady=3)
        tk.Label(rp_f, text="🔁 Repetition Penalty (1.0=выкл):", bg="#111111", fg="#cccccc").pack(side=tk.LEFT, padx=4)
        self.rep_pen_var = tk.DoubleVar(value=1.0)
        tk.Scale(rp_f, from_=1.0, to=3.0, resolution=0.05, orient=tk.HORIZONTAL,
                 variable=self.rep_pen_var, length=200, showvalue=True,
                 bg="#111111", fg="#00ff00", troughcolor="#222222",
                 highlightthickness=0).pack(side=tk.LEFT, padx=4)

        # EOS токены
        eos_f = tk.Frame(f, bg="#111111")
        eos_f.pack(fill=tk.X, padx=8, pady=3)
        tk.Label(eos_f, text="🛑 EOS строки (через запятую):", bg="#111111", fg="#cccccc").pack(side=tk.LEFT, padx=4)
        self.eos_var = tk.StringVar(value="")
        tk.Entry(eos_f, textvariable=self.eos_var, width=30,
                 bg="#1a1a1a", fg="#ff6666", insertbackground="#ff6666",
                 font=("Consolas", 10)).pack(side=tk.LEFT, padx=4)
        tk.Label(eos_f, text="пример: <|end|>,###", bg="#111111",
                 fg="#444444", font=("Consolas", 8)).pack(side=tk.LEFT)

        # Кнопка сброса
        r_f = tk.Frame(f, bg="#111111")
        r_f.pack(fill=tk.X, padx=8, pady=4)
        self._btn(r_f, "↺ Сброс настроек", self.reset_settings, "#7f8c8d").pack(side=tk.LEFT)

    # ----- Вкладка: SFT -----
    def _build_sft_tab(self, f):
        tk.Label(f, text="💬 Supervised Fine-Tuning (разговорная модель)",
                 bg="#111111", fg="#f39c12", font=("Consolas", 11, "bold")).pack(padx=8, pady=8, anchor=tk.W)

        tk.Label(f, text=(
            "SFT запускается ПОСЛЕ обучения базовой модели.\n"
            "Формат датасета — JSONL: каждая строка: {\"user\": \"...\", \"assistant\": \"...\"}\n"
            "Токены разделения: <|user|> ... <|assistant|> ... <|end|>"
        ), bg="#111111", fg="#888888", font=("Consolas", 9), justify=tk.LEFT).pack(padx=10, anchor=tk.W)

        # Путь к SFT датасету
        sf_f = tk.Frame(f, bg="#111111")
        sf_f.pack(fill=tk.X, padx=8, pady=6)
        tk.Label(sf_f, text="JSONL файл:", bg="#111111", fg="#cccccc").pack(side=tk.LEFT, padx=4)
        self.sft_path_var = tk.StringVar(value="")
        tk.Entry(sf_f, textvariable=self.sft_path_var, width=35,
                 bg="#1a1a1a", fg="#00ff00", insertbackground="#00ff00").pack(side=tk.LEFT, padx=4)
        self._btn(sf_f, "📂 Выбрать", self._browse_sft, "#555555").pack(side=tk.LEFT)

        # SFT LR
        lr_f = tk.Frame(f, bg="#111111")
        lr_f.pack(fill=tk.X, padx=8, pady=3)
        tk.Label(lr_f, text="LR для SFT:", bg="#111111", fg="#cccccc").pack(side=tk.LEFT, padx=4)
        self.sft_lr_var = tk.StringVar(value="1e-4")
        tk.Entry(lr_f, textvariable=self.sft_lr_var, width=10,
                 bg="#1a1a1a", fg="#00ff00", insertbackground="#00ff00").pack(side=tk.LEFT, padx=4)

        # SFT Iters
        si_f = tk.Frame(f, bg="#111111")
        si_f.pack(fill=tk.X, padx=8, pady=3)
        tk.Label(si_f, text="Кол-во шагов SFT:", bg="#111111", fg="#cccccc").pack(side=tk.LEFT, padx=4)
        self.sft_iters_var = tk.IntVar(value=2000)
        tk.Spinbox(si_f, from_=100, to=50000, increment=100,
                   textvariable=self.sft_iters_var, width=8,
                   bg="#1a1a1a", fg="#00ff00").pack(side=tk.LEFT, padx=4)

        self._btn(f, "💬 Запустить SFT", self.run_sft, "#8e44ad").pack(padx=8, pady=8, anchor=tk.W)

        tk.Label(f, text=(
            "Чтобы создать JSONL датасет:\n"
            "  [{\"user\": \"Что такое Т-34?\", \"assistant\": \"Т-34 — советский средний танк...\"}]\n"
            "Сохрани как .jsonl (одна пара на строку)."
        ), bg="#111111", fg="#555555", font=("Consolas", 8), justify=tk.LEFT).pack(padx=10, anchor=tk.W)

    # ----- Вкладка: Конфиг -----
    def _build_config_tab(self, f):
        tk.Label(f, text="⚙️ Гиперпараметры модели (применяются при следующем обучении)",
                 bg="#111111", fg="#f39c12", font=("Consolas", 10, "bold")).pack(padx=8, pady=8, anchor=tk.W)

        def row(parent, label, var, from_=None, to=None, step=None, is_str=False):
            rf = tk.Frame(parent, bg="#111111")
            rf.pack(fill=tk.X, padx=8, pady=2)
            tk.Label(rf, text=label, bg="#111111", fg="#cccccc", width=22, anchor=tk.W).pack(side=tk.LEFT)
            if is_str:
                e = tk.Entry(rf, textvariable=var, width=12, bg="#1a1a1a", fg="#00ff00",
                             insertbackground="#00ff00")
                e.pack(side=tk.LEFT, padx=4)
            else:
                sp = tk.Spinbox(rf, from_=from_, to=to, increment=step, textvariable=var,
                                width=10, bg="#1a1a1a", fg="#00ff00")
                sp.pack(side=tk.LEFT, padx=4)
            return rf

        self.cfg_batch   = tk.IntVar(value=config.batch_size)
        self.cfg_block   = tk.IntVar(value=config.block_size)
        self.cfg_iters   = tk.IntVar(value=config.max_iters)
        self.cfg_lr      = tk.StringVar(value=str(config.learning_rate))
        self.cfg_embd    = tk.IntVar(value=config.n_embd)
        self.cfg_head    = tk.IntVar(value=config.n_head)
        self.cfg_layer   = tk.IntVar(value=config.n_layer)
        self.cfg_dropout = tk.DoubleVar(value=config.dropout)

        row(f, "Batch size:",       self.cfg_batch, 1, 256, 1)
        row(f, "Block size:",       self.cfg_block, 32, 2048, 32)
        row(f, "Max iters:",        self.cfg_iters, 100, 200000, 100)
        row(f, "Learning rate:",    self.cfg_lr, is_str=True)
        row(f, "n_embd:",           self.cfg_embd, 64, 2048, 64)
        row(f, "n_head:",           self.cfg_head, 1, 32, 1)
        row(f, "n_layer:",          self.cfg_layer, 1, 24, 1)
        row(f, "Dropout:",          self.cfg_dropout, 0.0, 0.9, 0.05)

        self._btn(f, "✅ Применить конфиг", self._apply_config, "#27ae60").pack(padx=8, pady=8, anchor=tk.W)

    # =========================================================
    # Утилиты
    # =========================================================
    def _btn(self, parent, text, cmd, bg):
        return tk.Button(parent, text=text, command=cmd,
                         bg=bg, fg="white", font=("Arial", 9, "bold"),
                         relief=tk.FLAT, padx=6, pady=4, cursor="hand2")

    def print_log(self, txt, level="INFO"):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log.insert(tk.END, f"[{ts}] {txt}\n")
        self.log.see(tk.END)
        self.root.update()

    def update_data_info(self):
        bin_path = os.path.join(SCRIPT_DIR, 'train_data.bin')
        if os.path.exists(bin_path):
            size = os.path.getsize(bin_path) / 1024 / 1024
            self.data_info.config(text=f"✅ train_data.bin: {size:.2f} MB", fg="#27ae60")
        else:
            self.data_info.config(text="❌ Нет данных", fg="#e74c3c")

    def _get_n_tokens(self):
        """Возвращает число токенов из bin файла."""
        bin_path = os.path.join(SCRIPT_DIR, 'train_data.bin')
        if not os.path.exists(bin_path):
            return 0
        # Новый формат: dict с 'tokens' и 'mask'
        # Размер файла / 2 (tokens + mask, оба примерно по 8 байт на элемент)
        # Точнее — просто делим размер на 9 (8 байт long + 1 байт bool)
        return os.path.getsize(bin_path) // 9

    def _apply_config(self):
        try:
            config.batch_size    = self.cfg_batch.get()
            config.block_size    = self.cfg_block.get()
            config.max_iters     = self.cfg_iters.get()
            config.learning_rate = float(self.cfg_lr.get())
            config.n_embd        = self.cfg_embd.get()
            config.n_head        = self.cfg_head.get()
            config.n_layer       = self.cfg_layer.get()
            config.dropout       = self.cfg_dropout.get()
            self.print_log("✅ Конфиг применён!")
        except Exception as e:
            self.print_log(f"❌ Ошибка конфига: {e}")

    def _draw_loss_graph(self):
        """Рисует loss на Canvas."""
        c = self.loss_canvas
        c.delete("all")
        w = c.winfo_width()
        h = c.winfo_height()
        if len(self.loss_history) < 2 or w < 10:
            return

        vals = self.loss_history[-min(len(self.loss_history), 200):]
        mn, mx = min(vals), max(vals)
        if mx == mn:
            return

        def y(v):
            return h - 5 - int((v - mn) / (mx - mn) * (h - 10))

        step = w / (len(vals) - 1)
        pts = [(i * step, y(v)) for i, v in enumerate(vals)]

        for i in range(len(pts) - 1):
            c.create_line(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1],
                          fill="#00ff00", width=1)

        # Последнее значение
        c.create_text(w - 5, 8, text=f"{vals[-1]:.4f}", fill="#ffff00",
                      font=("Consolas", 8), anchor=tk.NE)

    def _parse_eos_tokens(self):
        """Парсит строку EOS токенов в список int."""
        raw = self.eos_var.get().strip()
        if not raw:
            return []
        ids = []
        for s in raw.split(","):
            s = s.strip()
            if s:
                ids.extend(self.enc.encode_ordinary(s))
        return list(set(ids))

    def _browse_sft(self):
        path = filedialog.askopenfilename(filetypes=[("JSONL", "*.jsonl"), ("All", "*.*")])
        if path:
            self.sft_path_var.set(path)

    def _on_mode_change(self):
        self._update_epoch_info()

    def _set_epochs(self, n):
        self.train_mode.set("epochs")
        self.epochs_spin_var.set(n)
        self._update_epoch_info()

    def _update_epoch_info(self):
        n_tokens = self._get_n_tokens()
        if n_tokens == 0:
            self.epoch_info_var.set("(нет данных)")
            return
        try:
            iters_per_epoch = max(1, n_tokens // (config.batch_size * config.block_size))
            if self.train_mode.get() == "epochs":
                n_ep = self.epochs_spin_var.get()
                total = n_ep * iters_per_epoch
                self.epoch_info_var.set(f"= {total:,} iter  (1ep={iters_per_epoch:,})")
            else:
                n_it = self.iters_spin_var.get()
                epochs = n_it / iters_per_epoch
                self.epoch_info_var.set(f"≈ {epochs:.1f} эпох  (1ep={iters_per_epoch:,} iter)")
        except Exception:
            self.epoch_info_var.set("")

    def _get_total_iters(self):
        n_tokens = self._get_n_tokens()
        if self.train_mode.get() == "iters":
            return self.iters_spin_var.get()
        iters_per_epoch = max(1, n_tokens // (config.batch_size * config.block_size)) if n_tokens else 1000
        return self.epochs_spin_var.get() * iters_per_epoch

    def reset_settings(self):
        self.temp_var.set("0.8")
        self.tokens_var.set(300)
        self.topk_var.set(0)
        self.topp_var.set(1.0)
        self.rep_pen_var.set(1.0)
        self.eos_var.set("")
        self.print_log("⚙️ Настройки генерации сброшены")

    # =========================================================
    # ПАРСИНГ
    # =========================================================
    def fetch(self):
        def work():
            self.status_var.set("🟡 Парсинг...")
            tank_urls = [
                # СССР/Россия
                "https://wiki.warthunder.ru/T-80BVM", "https://wiki.warthunder.ru/T-90A",
                "https://wiki.warthunder.ru/T-72B3",  "https://wiki.warthunder.ru/T-64B",
                "https://wiki.warthunder.ru/T-62",    "https://wiki.warthunder.ru/T-55A",
                "https://wiki.warthunder.ru/T-44-100","https://wiki.warthunder.ru/T-34-85",
                "https://wiki.warthunder.ru/IS-2",    "https://wiki.warthunder.ru/IS-3",
                "https://wiki.warthunder.ru/IS-4M",   "https://wiki.warthunder.ru/IS-7",
                "https://wiki.warthunder.ru/KV-1",    "https://wiki.warthunder.ru/KV-2",
                "https://wiki.warthunder.ru/SU-100",  "https://wiki.warthunder.ru/SU-122",
                "https://wiki.warthunder.ru/ZSU-23-4",
                # Германия
                "https://wiki.warthunder.ru/Leopard_2A6","https://wiki.warthunder.ru/Leopard_2A5",
                "https://wiki.warthunder.ru/Leopard_2A4","https://wiki.warthunder.ru/Leopard_1",
                "https://wiki.warthunder.ru/Tiger_H1",   "https://wiki.warthunder.ru/Tiger_II",
                "https://wiki.warthunder.ru/Panther_G",  "https://wiki.warthunder.ru/Panther_F",
                "https://wiki.warthunder.ru/Jagdpanther","https://wiki.warthunder.ru/Ferdinand",
                "https://wiki.warthunder.ru/Jagdtiger",  "https://wiki.warthunder.ru/StuG_III",
                "https://wiki.warthunder.ru/Flakpanzer",
                # США
                "https://wiki.warthunder.ru/M1A2_SEP_V2","https://wiki.warthunder.ru/M1A1_Abrams",
                "https://wiki.warthunder.ru/M1_Abrams",  "https://wiki.warthunder.ru/M60A3_TTS",
                "https://wiki.warthunder.ru/M60A1",      "https://wiki.warthunder.ru/M48A1",
                "https://wiki.warthunder.ru/M47",        "https://wiki.warthunder.ru/M4_Sherman",
                "https://wiki.warthunder.ru/M4A3E8",     "https://wiki.warthunder.ru/M18_Hellcat",
                "https://wiki.warthunder.ru/M36_Jackson","https://wiki.warthunder.ru/M103",
                "https://wiki.warthunder.ru/M163_VADS",
                # Британия
                "https://wiki.warthunder.ru/Challenger_2","https://wiki.warthunder.ru/Challenger_1",
                "https://wiki.warthunder.ru/Chieftain_Mk10","https://wiki.warthunder.ru/Conqueror",
                "https://wiki.warthunder.ru/Centurion_Mk10","https://wiki.warthunder.ru/Black_Prince",
                "https://wiki.warthunder.ru/Churchill_VII",
                # Франция
                "https://wiki.warthunder.ru/Leclerc",    "https://wiki.warthunder.ru/Leclerc_S2",
                "https://wiki.warthunder.ru/AMX-40",     "https://wiki.warthunder.ru/AMX-30",
                "https://wiki.warthunder.ru/AMX-13",     "https://wiki.warthunder.ru/AMX-50",
                # Япония
                "https://wiki.warthunder.ru/Type_10",    "https://wiki.warthunder.ru/Type_90",
                "https://wiki.warthunder.ru/Type_74",    "https://wiki.warthunder.ru/Type_61",
                "https://wiki.warthunder.ru/ST-A1",
                # Китай
                "https://wiki.warthunder.ru/ZTZ99A",     "https://wiki.warthunder.ru/ZTZ96A",
                "https://wiki.warthunder.ru/ZTZ88B",     "https://wiki.warthunder.ru/Type_69",
                "https://wiki.warthunder.ru/Type_59",
                # Израиль
                "https://wiki.warthunder.ru/Merkava_Mk4","https://wiki.warthunder.ru/Merkava_Mk3",
                "https://wiki.warthunder.ru/Merkava_Mk2","https://wiki.warthunder.ru/Merkava_Mk1",
                "https://wiki.warthunder.ru/Shot_Kal",
                # Италия
                "https://wiki.warthunder.ru/Ariete",     "https://wiki.warthunder.ru/Ariete_PSO",
                "https://wiki.warthunder.ru/OF-40",      "https://wiki.warthunder.ru/M26_Pershing",
                # Швеция
                "https://wiki.warthunder.ru/Strv_122",   "https://wiki.warthunder.ru/Strv_103",
                "https://wiki.warthunder.ru/Strv_81",    "https://wiki.warthunder.ru/IKV_91",
            ]

            all_text = ""
            total_pages = 0
            headers = {'User-Agent': 'Mozilla/5.0'}
            use_sep = self.use_s_text.get()

            for i, url in enumerate(tank_urls):
                try:
                    self.print_log(f"📡 [{i+1}/{len(tank_urls)}] {url}")
                    time.sleep(1)
                    response = requests.get(url, timeout=10, headers=headers)
                    soup = BeautifulSoup(response.text, 'html.parser')

                    # Разделитель между документами
                    if use_sep and all_text:
                        all_text += f"<end>\n{S_TEXT_TOKEN}\n"

                    title = soup.find('h1')
                    if title:
                        all_text += f"\n\n=== {title.get_text()} ===\n\n"

                    for p in soup.find_all('p'):
                        text = p.get_text().strip()
                        if len(text) > 30:
                            all_text += text + ""

                    total_pages += 1
                    self.print_log(f"   ✅ Успешно")

                except Exception as e:
                    self.print_log(f"   ⚠️ Ошибка: {str(e)[:60]}")

            if all_text:
                fname = f"input_wiki_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                save_path = os.path.join(SCRIPT_DIR, fname)
                with open(save_path, "w", encoding="utf-8") as ff:
                    ff.write(all_text)
                self.print_log(f"✅ ГОТОВО! Файл: {fname}")
                self.print_log(f"📏 Размер: {len(all_text)/1024:.1f} KB")
                self.update_data_info()
            else:
                self.print_log("❌ Нет данных")

            self.status_var.set("🟢 Готов")

        threading.Thread(target=work, daemon=True).start()

    # =========================================================
    # ТОКЕНИЗАЦИЯ
    # =========================================================
    def tokenize(self):
        self.print_log("="*50)
        self.print_log("🔨 ТОКЕНИЗАЦИЯ")

        text_files = [f for f in os.listdir(SCRIPT_DIR) if f.endswith('.txt')]
        if not text_files:
            self.print_log("❌ Нет .txt файлов!")
            return

        combined_text = ""
        use_sep = self.use_s_text.get()

        for f_name in text_files:
            try:
                with open(os.path.join(SCRIPT_DIR, f_name), 'r', encoding='utf-8') as f:
                    chunk = f.read()
                    if use_sep and combined_text:
                        combined_text += f"\n{S_TEXT_TOKEN}\n"
                    combined_text += chunk
            except Exception as e:
                self.print_log(f"⚠️ Ошибка: {f_name} - {e}")

        self.print_log("⏳ Токенизация...")

        # Токенизируем по частям — между <s_text> токенизируем отдельно
        # чтобы знать точные границы документов
        sep = S_TEXT_TOKEN
        parts = combined_text.split(sep)
        self.print_log(f"📄 Документов: {len(parts)}")

        tokens = []
        # mask: 0 = нормальный токен (учим), 1 = граница (не учим loss)
        mask = []
        sep_ids = self.enc.encode_ordinary(sep)  # токены самого разделителя

        for idx, part in enumerate(parts):
            part_tokens = self.enc.encode_ordinary(part)
            tokens.extend(part_tokens)
            mask.extend([0] * len(part_tokens))
            # Добавляем разделитель между частями (но не после последней)
            if idx < len(parts) - 1:
                tokens.extend(sep_ids)
                mask.extend([1] * len(sep_ids))  # эти позиции — игнор при loss

        if config.device == 'cuda':
            max_tokens = 111_000_000
        else:
            max_tokens = 300_000
        self.print_log(f"ℹ️ Лимит: {max_tokens:,} токенов")

        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
            mask   = mask[:max_tokens]
            self.print_log(f"⚠️ Обрезано до {max_tokens:,}")

        sep_count = mask.count(1)
        self.print_log(f"🔒 Замаскировано граничных токенов: {sep_count:,}")

        save_path = os.path.join(SCRIPT_DIR, 'train_data.bin')
        torch.save({
            'tokens': torch.tensor(tokens, dtype=torch.long),
            'mask':   torch.tensor(mask,   dtype=torch.bool),
        }, save_path)
        self.print_log(f"✅ Токенов: {len(tokens):,}")
        self.update_data_info()

    # =========================================================
    # ОБУЧЕНИЕ
    # =========================================================
    def train(self):
        if self.training:
            self.print_log("⚠️ Обучение уже идёт!")
            return

        bin_path = os.path.join(SCRIPT_DIR, 'train_data.bin')
        if not os.path.exists(bin_path):
            self.print_log("❌ Нет train_data.bin!")
            return

        self._apply_config()
        self._update_epoch_info()

        def work():
            self.training = True
            self.train_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.status_var.set("🟡 ОБУЧЕНИЕ...")
            self.loss_history.clear()

            data = torch.load(bin_path, weights_only=True)
            tokens = data['tokens'].to(config.device)
            # mask: True = граница <s_text>, loss на этих позициях = игнор
            sep_mask = data['mask'].to(config.device)
            n_tokens = len(tokens)
            model_path = os.path.join(SCRIPT_DIR, 'tank_model.pth')

            if self.model is None:
                self.model = TankGPT(self.enc.n_vocab, dropout=config.dropout).to(config.device)
                if os.path.exists(model_path):
                    self.model.load_state_dict(torch.load(model_path, map_location=config.device, weights_only=True))
                    self.print_log("✅ Загружена предыдущая модель")

            # torch.compile (PyTorch 2.0+)
            compiled_model = self.model
            if self.use_compile.get():
                try:
                    compiled_model = torch.compile(self.model)
                    self.print_log("⚡ torch.compile() активен")
                except Exception as e:
                    self.print_log(f"⚠️ torch.compile недоступен: {e}")

            # Параметры
            total_iters     = self._get_total_iters()
            grad_accum      = self.grad_accum_var.get()
            checkpoint_ep   = self.checkpoint_var.get()
            use_amp         = self.use_amp.get() and config.device == 'cuda'
            use_scheduler   = self.use_scheduler.get()

            # Считаем иters_per_epoch для чекпоинтов
            iters_per_epoch = max(1, n_tokens // (config.batch_size * config.block_size))

            optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate,
                                          weight_decay=0.1, betas=(0.9, 0.95))

            # Cosine LR scheduler с warmup
            warmup_iters = min(100, total_iters // 10)
            if use_scheduler:
                def lr_lambda(step):
                    if step < warmup_iters:
                        return step / max(1, warmup_iters)
                    progress = (step - warmup_iters) / max(1, total_iters - warmup_iters)
                    return 0.1 + 0.9 * 0.5 * (1 + __import__('math').cos(__import__('math').pi * progress))
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            else:
                scheduler = None

            # AMP scaler
            scaler = torch.amp.GradScaler('cuda') if use_amp else None

            self.progress['maximum'] = total_iters
            self.progress['value']   = 0
            self.start_time = time.time()

            def get_batch():
                ix = torch.randint(n_tokens - config.block_size, (config.batch_size,))
                x = torch.stack([tokens[i:i+config.block_size]     for i in ix])
                y = torch.stack([tokens[i+1:i+config.block_size+1] for i in ix])
                # Маскируем таргеты на границах <s_text> — loss не считается
                m = torch.stack([sep_mask[i+1:i+config.block_size+1] for i in ix])
                y[m] = -100  # cross_entropy игнорирует -100
                return x, y

            amp_str = "AMP" if use_amp else ""
            sch_str = "Cosine" if use_scheduler else ""
            cmp_str = "Compile" if self.use_compile.get() else ""
            flags = " | ".join(filter(None, [amp_str, sch_str, cmp_str]))
            self.print_log(f"🚀 СТАРТ | {config.device.upper()} | {total_iters:,} iter | {flags}")
            self.print_log(f"   1 эпоха = {iters_per_epoch:,} итераций")

            compiled_model.train()
            optimizer.zero_grad()

            for i in range(total_iters):
                if not self.training:
                    break

                xb, yb = get_batch()

                if use_amp:
                    with torch.amp.autocast('cuda'):
                        _, loss = compiled_model(xb, yb)
                    loss = loss / grad_accum
                    scaler.scale(loss).backward()
                    if (i + 1) % grad_accum == 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    _, loss = compiled_model(xb, yb)
                    loss = loss / grad_accum
                    loss.backward()
                    if (i + 1) % grad_accum == 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        optimizer.step()
                        optimizer.zero_grad()

                if scheduler:
                    scheduler.step()

                real_loss = loss.item() * grad_accum
                self.loss_history.append(real_loss)

                if i % 100 == 0:
                    elapsed  = time.time() - self.start_time
                    cur_lr   = optimizer.param_groups[0]['lr']
                    cur_ep   = i / iters_per_epoch
                    self.epoch_label_var.set(
                        f"Шаг {i:,}/{total_iters:,} | loss={real_loss:.4f} | ep={cur_ep:.2f} | lr={cur_lr:.2e} | {elapsed:.0f}с"
                    )
                    self.print_log(
                        f"Шаг {i:>7,}: loss={real_loss:.4f} | ep={cur_ep:.2f} | lr={cur_lr:.2e} | {elapsed:.0f}с"
                    )
                    self._draw_loss_graph()

                # Checkpoint каждые N эпох
                cur_epoch = i // iters_per_epoch
                prev_epoch = (i - 1) // iters_per_epoch if i > 0 else -1
                if cur_epoch != prev_epoch and cur_epoch > 0 and cur_epoch % checkpoint_ep == 0:
                    ck_path = os.path.join(SCRIPT_DIR, f'checkpoint_ep{cur_epoch}.pth')
                    torch.save(self.model.state_dict(), ck_path)
                    self.print_log(f"💾 Чекпоинт: checkpoint_ep{cur_epoch}.pth")

                self.progress['value'] = i + 1
                self.root.update()

            torch.save(self.model.state_dict(), model_path)
            self._draw_loss_graph()
            total_ep = total_iters / iters_per_epoch
            self.print_log(f"✅ ОБУЧЕНИЕ ЗАВЕРШЕНО! {total_iters:,} итераций = {total_ep:.1f} эпох")

            self.training = False
            self.train_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self.status_var.set("🟢 Готов")

        threading.Thread(target=work, daemon=True).start()

    def stop_training(self):
        self.training = False
        self.print_log("⏹ Останавливаем...")

    # =========================================================
    # ГЕНЕРАЦИЯ
    # =========================================================
    def test_gen(self):
        model_path = os.path.join(SCRIPT_DIR, 'tank_model.pth')

        if self.model is None:
            if os.path.exists(model_path):
                self.model = TankGPT(self.enc.n_vocab, dropout=0.0).to(config.device)
                self.model.load_state_dict(torch.load(model_path, map_location=config.device, weights_only=True))
                self.print_log("✅ Модель загружена")
            else:
                self.print_log("❌ Модель не найдена!")
                return

        self.model.eval()
        prompt = self.prompt_entry.get()

        # Температура — парсим вручную
        try:
            temp = float(self.temp_var.get())
            temp = max(0.0, min(100.0, temp))
        except ValueError:
            self.print_log("⚠️ Некорректная температура, ставлю 0.8")
            temp = 0.8

        max_tokens     = self.tokens_var.get()
        top_k          = self.topk_var.get()
        top_p          = self.topp_var.get()
        rep_pen        = self.rep_pen_var.get()
        eos_token_ids  = self._parse_eos_tokens()

        self.print_log("="*50)
        self.print_log(f"❓ Запрос: {prompt}")
        self.print_log(f"🌡️ temp={temp} | top_k={top_k} | top_p={top_p} | rep_pen={rep_pen} | max_tok={max_tokens}")
        if eos_token_ids:
            self.print_log(f"🛑 EOS ids: {eos_token_ids}")

        def stream_work():
            try:
                start_ids = self.enc.encode_ordinary(prompt)
                idx = torch.tensor([start_ids], dtype=torch.long, device=config.device)

                # Пишем метку ответа один раз
                ts = datetime.now().strftime("%H:%M:%S")
                self.log.insert(tk.END, f"[{ts}] 🤖 ")
                self.log.see(tk.END)
                self.root.update()

                generated_ids = []

                with torch.no_grad():
                    for token_id in self.model.generate_stream(
                        idx, max_tokens,
                        temperature=temp, top_k=top_k, top_p=top_p,
                        repetition_penalty=rep_pen, eos_token_ids=eos_token_ids
                    ):
                        generated_ids.append(token_id)

                        # Декодируем весь накопленный текст и берём только хвост
                        # (tiktoken может менять декодировку при многобайтовых символах)
                        full_text = self.enc.decode(generated_ids)
                        prev_text = self.enc.decode(generated_ids[:-1]) if len(generated_ids) > 1 else ""
                        new_piece = full_text[len(prev_text):]

                        if new_piece:
                            self.log.insert(tk.END, new_piece)
                            self.log.see(tk.END)
                            self.root.update()

                # Перевод строки после конца генерации
                self.log.insert(tk.END, "\n")
                self.log.see(tk.END)
                self.status_var.set("🟢 Готов")
                self.root.update()

            except Exception as e:
                self.print_log(f"❌ Ошибка: {e}")

        self.status_var.set("🟡 Генерация...")
        threading.Thread(target=stream_work, daemon=True).start()

    # =========================================================
    # SFT
    # =========================================================
    def run_sft(self):
        """Supervised Fine-Tuning на JSONL датасете."""
        model_path = os.path.join(SCRIPT_DIR, 'tank_model.pth')
        sft_path   = self.sft_path_var.get().strip()

        if not os.path.exists(model_path):
            self.print_log("❌ Сначала обучи базовую модель!")
            return
        if not sft_path or not os.path.exists(sft_path):
            self.print_log("❌ Укажи JSONL файл!")
            return

        def work():
            self.status_var.set("🟡 SFT...")
            self.print_log("="*50)
            self.print_log("💬 SFT ЗАПУЩЕН")

            # Загружаем базовую модель
            if self.model is None:
                self.model = TankGPT(self.enc.n_vocab, dropout=config.dropout).to(config.device)
            self.model.load_state_dict(torch.load(model_path, map_location=config.device))

            # Читаем JSONL
            pairs = []
            with open(sft_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            obj = json.loads(line)
                            pairs.append(obj)
                        except:
                            pass
            self.print_log(f"📦 Загружено пар: {len(pairs)}")

            # Формируем токены в формате диалога
            # <|user|> вопрос <|assistant|> ответ <|end|>
            all_tokens = []
            for p in pairs:
                text = f"<|user|> {p.get('user','')} <|assistant|> {p.get('assistant','')} <|end|>\n"
                all_tokens.extend(self.enc.encode_ordinary(text))

            if not all_tokens:
                self.print_log("❌ Датасет пуст!")
                self.status_var.set("🟢 Готов")
                return

            data = torch.tensor(all_tokens, dtype=torch.long).to(config.device)
            self.print_log(f"📝 Токенов SFT: {len(all_tokens):,}")

            try:
                sft_lr = float(self.sft_lr_var.get())
            except:
                sft_lr = 1e-4

            sft_iters = self.sft_iters_var.get()
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=sft_lr)
            self.model.train()

            self.progress['maximum'] = sft_iters
            self.progress['value']   = 0
            self.loss_history.clear()

            def get_batch():
                if len(data) <= config.block_size:
                    ix = torch.zeros(1, dtype=torch.long)
                else:
                    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
                x = torch.stack([data[i:i+config.block_size]     for i in ix]).to(config.device)
                y = torch.stack([data[i+1:i+config.block_size+1] for i in ix]).to(config.device)
                return x, y

            for i in range(sft_iters):
                if not self.training and i > 0:
                    break
                xb, yb = get_batch()
                _, loss = self.model(xb, yb)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                self.loss_history.append(loss.item())
                if i % 100 == 0:
                    self.print_log(f"[SFT] Шаг {i}: loss={loss.item():.4f}")
                    self._draw_loss_graph()

                self.progress['value'] = i + 1
                self.root.update()

            sft_save = os.path.join(SCRIPT_DIR, 'tank_model_sft.pth')
            torch.save(self.model.state_dict(), sft_save)
            self.print_log(f"✅ SFT ЗАВЕРШЁН! Сохранено: tank_model_sft.pth")
            self.status_var.set("🟢 Готов")

        threading.Thread(target=work, daemon=True).start()


# =========================================================
# ЗАПУСК
# =========================================================
if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style()
    style.theme_use('clam')
    style.configure('TNotebook', background='#111111', borderwidth=0)
    style.configure('TNotebook.Tab', background='#222222', foreground='#aaaaaa',
                    padding=[10, 4], font=('Consolas', 9))
    style.map('TNotebook.Tab', background=[('selected', '#0d0d0d')],
              foreground=[('selected', '#00ff00')])
    style.configure('Horizontal.TProgressbar', troughcolor='#222222',
                    background='#00ff00', thickness=8)
    app = UltimateStation(root)
    root.mainloop()
