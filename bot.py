import os
import subprocess
import sys

# 1. تثبيت المكتبات اللازمة تلقائياً إن لم تكن موجودة
required_packages = ["requests", "pandas", "numpy", "matplotlib", "scikit-learn", "yfinance"]
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ==========================================
# إعدادات التليجرام وإدارة المخاطر (تسحب من النظام تلقائياً)
# ==========================================
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('CHAT_ID')

def send_telegram_message(message):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print("خطأ في إرسال إشعار تليجرام:", e)

SYMBOL = "BNB-USD"
INTERVAL = "1h"
INITIAL_CAPITAL = 100.0
STOP_LOSS_PCT = 0.02
TAKE_PROFIT_PCT = 0.04
TRAIL_STOP_PCT = 0.015
MAX_DAILY_LOSS = 0.03

print("=" * 60)
print("🤖 BNB AI TRADING AGENT - RUNNING...")
print("=" * 60)

send_telegram_message("🚀 تم تشغيل وكيل تداول BNB آلياً عبر جيت هاب بنجاح.")

# 2. جلب أحدث بيانات السوق
df = yf.download(SYMBOL, period="30d", interval=INTERVAL)

if isinstance(df.columns, pd.MultiIndex):
    df.columns = [col[0] for col in df.columns]
df = df.reset_index()
df.columns = [str(c).lower() for c in df.columns]
if 'date' in df.columns:
    df.rename(columns={'date': 'open_time'}, inplace=True)
elif 'datetime' in df.columns:
    df.rename(columns={'datetime': 'open_time'}, inplace=True)

# 3. حساب المؤشرات الفنية
df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()

delta = df['close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['rsi'] = 100 - (100 / (1 + rs))

df = df.dropna().reset_index(drop=True)

# 4. تدريب نموذج الذكاء الاصطناعي
df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
features = ['ema20', 'ema50', 'rsi']
X = df[features].iloc[:-1]
y = df['target'].iloc[:-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

df['ai_signal'] = model.predict(df[features])

# 5. محرك التداول وإدارة المخاطر
capital = INITIAL_CAPITAL
position = None
entry_price = 0
highest_price = 0.0
trades = []

daily_loss = 0.0
daily_block = False
current_day = None

for i in range(len(df) - 1):
    row = df.iloc[i]
    signal = row['ai_signal']
    current_price = row['close']
    row_date = row['open_time'].date() if hasattr(row['open_time'], 'date') else row['open_time']
    
    if current_day != row_date:
        current_day = row_date
        daily_loss = 0.0
        daily_block = False

    if daily_block:
        continue

    if position is None and signal == 1:
        position = "BUY"
        entry_price = current_price
        highest_price = current_price
        
        msg = f"🟢 **صفقة شراء جديدة (BNB)**\nالسعر: `{round(current_price, 2)}`\nالتاريخ: `{row_date}`"
        send_telegram_message(msg)
        
    elif position == "BUY":
        if current_price > highest_price:
            highest_price = current_price
            
        pnl_pct = (current_price - entry_price) / entry_price
        trail_drop = (current_price - highest_price) / highest_price
        
        if trail_drop <= -TRAIL_STOP_PCT or pnl_pct <= -STOP_LOSS_PCT or pnl_pct >= TAKE_PROFIT_PCT or signal == 0:
            trade_pnl = capital * pnl_pct
            capital += trade_pnl
            trades.append(trade_pnl)
            
            status_emoji = "✅ ربح" if trade_pnl > 0 else "❌ خسارة"
            msg = f"{status_emoji} **إغلاق صفقة BNB**\nالربح/الخسارة: `{round(trade_pnl, 2)} USDT`\nرأس المال الحالي: `{round(capital, 2)} USDT`"
            send_telegram_message(msg)
            
            if trade_pnl < 0:
                daily_loss += abs(trade_pnl)
                if daily_loss >= (INITIAL_CAPITAL * MAX_DAILY_LOSS):
                    daily_block = True
                    send_telegram_message("⚠️ **تحذير:** تم بلوغ حد الخسارة اليومي، تم إيقاف التداول مؤقتاً.")
                    
            position = None

print("=" * 30)
print(f"💰 رأس المال الأولي: {INITIAL_CAPITAL} USDT")
print(f"💵 رأس المال النهائي: {round(capital, 2)} USDT")
print(f"📊 عدد الصفقات المنفذة: {len(trades)}")
print("=" * 60)
