# Transaction Costs (Estimations based on Indian F&O rules)
TRANSACTION_COSTS = {
    'brokerage_per_order': 20.0,
    'stt_options_sell': 0.000625,  # 0.0625% on premium for selling options (NSE)
    'stt_futures': 0.000125,       # 0.0125% on sell side futures
    'exchange_txn_charge': 0.0005, # ~0.05% blended
    'gst': 0.18,                   # 18% on (brokerage + exchange charges)
    'stamp_duty': 0.00003,         # 0.003% on buy side
    'sebi_turnover': 0.000001      # 10 per crore
}

# Dhan API Endpoints / Mapping URLs
DHAN_INSTRUMENT_MASTER_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"
