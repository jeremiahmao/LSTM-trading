TIME_STEP = 120
NUM_FEATURES = 10

#trying to contain stocks that follow a normal growth and normal variation over time (not as high volatility), additionally IPO'd before 2013 June 1, since that is when we will start our data collection
#data collection for training data will end 2023 June 1
TRAINING_SYMBOLS = [
    # High-quality stocks with normal growth and variation
    "AAPL",   # Apple Inc.
    "ABBV",   # AbbVie Inc. (spun off from Abbott Laboratories in January 2013)
    "ACN",    # Accenture plc
    "ADBE",   # Adobe Inc.
    "AMD",    # Advanced Micro Devices Inc.
    "AMGN",   # Amgen Inc.
    "AMZN",   # Amazon.com Inc.
    "AXP",    # American Express Co.
    "BA",     # Boeing Co.
    "BAC",    # Bank of America Corp.
    "BBY",    # Best Buy (Retail, facing e-commerce challenges)
    "CAT",    # Caterpillar Inc.
    "COST",   # Costco Wholesale Corp.
    "CRM",    # Salesforce Inc.
    "CSCO",   # Cisco Systems Inc.
    "CVX",    # Chevron Corp.
    "DHR",    # Danaher Corp.
    "DIS",    # The Walt Disney Co.
    "DOW",    # Dow Inc.
    "GE",     # General Electric (Industrial conglomerate, restructuring)
    "GOOG",  # Alphabet Inc. (Google)
    "GS",     # Goldman Sachs Group Inc.
    "HD",     # Home Depot Inc.
    "HON",    # Honeywell International Inc.
    "IBM",    # International Business Machines Corp.
    "INTC",   # Intel Corp.
    "JNJ",    # Johnson & Johnson
    "KO",     # Coca-Cola Co.
    "LLY",    # Eli Lilly and Co.
    "MCD",    # McDonald's Corp.
    "META",   # Meta Platforms Inc. (Facebook, previously known as Facebook Inc.)
    "MMM",    # 3M Co.
    "MS",     # Morgan Stanley
    "MSFT",   # Microsoft Corp.
    "NFLX",   # Netflix Inc.
    "NKE",    # Nike Inc.
    "NVDA",   # NVIDIA Corp.
    "ORCL",   # Oracle Corp.
    "PFE",    # Pfizer Inc.
    "PG",     # Procter & Gamble Co.
    "TMO",    # Thermo Fisher Scientific Inc.
    "TRV",    # Travelers Companies Inc.
    "TSLA",   # Tesla Inc.
    "UNH",    # UnitedHealth Group Inc.
    "V",      # Visa Inc.
    "VZ",     # Verizon Communications Inc.
    "WBA",    # Walgreens Boots Alliance Inc.
    "WFC",    # Wells Fargo & Co.
    "WMT",    # Walmart Inc.

    # Consistently poorly performing stocks
    "CWH",    # Camping World Holdings Inc.
    "FUBO",   # fuboTV Inc.
    "GEO",    # GEO Group Inc.
    "GPRO",   # GoPro Inc.
    "MARA",   # Marathon Digital Holdings Inc.
    "MTCH",   # Match Group Inc.
    "NOK",    # Nokia Corporation
    "NTR",    # Nutrien Ltd.
    "SAVA",   # Cassava Sciences Inc.

    # Meme stocks
    "GME",    # GameStop Corp.
    "AMC",    # AMC Entertainment Holdings Inc.
    "BB",     # BlackBerry Limited
    "PLTR",   # Palantir Technologies Inc.

    # Cyclic stocks
    "GM",     # General Motors Company
    "DHI",    # D.R. Horton, Inc. (homebuilder)
    "LEN",    # Lennar Corporation (homebuilder)
    "NUE",    # Nucor Corporation (steel production)
    "X",      # United States Steel Corporation
    "DE",     # Deere & Company (agricultural equipment)
    "PCAR",   # PACCAR Inc. (truck manufacturer)
    "ETN"     # Eaton Corporation plc (industrial products)
]

TEST_SYMBOLS = ['NVDA']