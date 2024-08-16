CANDLES_TIME_STEP = 120
MERGED_TIME_STEP = 30
NUM_FEATURES = 8

MAX_REQUESTS_PER_MINUTE = 75

#trying to contain stocks that follow a normal growth and normal variation over time (not as high volatility), additionally IPO'd before 2013 June 1, since that is when we will start our data collection
#data collection for training data will end 2023 June 1
CANDLES_TRAINING_SYMBOLS = [
    # High-quality stocks with normal growth and variation
    "AAPL",   # Apple Inc.
    "ABBV",   # AbbVie Inc.
    "ACN",    # Accenture plc
    "ADBE",   # Adobe Inc.
    "AMD",    # Advanced Micro Devices Inc.
    "AMGN",   # Amgen Inc.
    "AMZN",   # Amazon.com Inc.
    "AIG",    # American International Group Inc.
    "AXP",    # American Express Co.
    "BA",     # Boeing Co.
    "BAC",    # Bank of America Corp.
    "BBY",    # Best Buy
    "BMY",    # Bristol-Myers Squibb Co.
    "C",      # Citigroup Inc.
    "CAT",    # Caterpillar Inc.
    "CME",    # CME Group Inc.
    "COST",   # Costco Wholesale Corp.
    "CRM",    # Salesforce Inc.
    "CSCO",   # Cisco Systems Inc.
    "CVX",    # Chevron Corp.
    "D",      # Dominion Energy Inc.
    "DHR",    # Danaher Corp.
    "DIS",    # The Walt Disney Co.
    "DOW",    # Dow Inc.
    "ECL",    # Ecolab Inc.
    "ETSY",   # Etsy Inc.
    "EXC",    # Exelon Corporation
    "F",      # Ford Motor Company
    "FIS",    # Fidelity National Information Services Inc.
    "FISV",   # Fiserv Inc.
    "GILD",   # Gilead Sciences Inc.
    "GPN",    # Global Payments Inc.
    "GOOG",   # Alphabet Inc.
    "GS",     # Goldman Sachs Group Inc.
    "HCA",    # HCA Healthcare Inc.
    "HES",    # Hess Corp.
    "HSY",    # The Hershey Company
    "IBM",    # International Business Machines Corp.
    "INTC",   # Intel Corp.
    "ISRG",   # Intuitive Surgical Inc.
    "JNJ",    # Johnson & Johnson
    "KO",     # Coca-Cola Co.
    "KHC",    # Kraft Heinz Co.
    "LLY",    # Eli Lilly and Co.
    "LMT",    # Lockheed Martin Corp.
    "LNC",    # Lincoln National Corporation
    "MAR",    # Marriott International Inc.
    "MDT",    # Medtronic PLC
    "MCD",    # McDonald's Corp.
    "MCK",    # McKesson Corporation
    "META",   # Meta Platforms Inc.
    "MMM",    # 3M Co.
    "MRO",    # Marathon Oil Corporation
    "MS",     # Morgan Stanley
    "MSCI",   # MSCI Inc.
    "MSFT",   # Microsoft Corp.
    "NDAQ",   # Nasdaq Inc.
    "NKE",    # Nike Inc.
    "NOC",    # Northrop Grumman Corp.
    "NVDA",   # NVIDIA Corp.
    "NVR",    # NVR Inc.
    "ORCL",   # Oracle Corp.
    "PFE",    # Pfizer Inc.
    "PG",     # Procter & Gamble Co.
    "PGR",    # Progressive Corporation
    "PCAR",   # PACCAR Inc.
    "PXD",    # Pioneer Natural Resources Company
    "QCOM",   # Qualcomm Inc.
    "RBLX",   # Roblox Corp.
    "RCL",    # Royal Caribbean Group
    "RTX",    # Raytheon Technologies Corp.
    "SBUX",   # Starbucks Corporation
    "SRE",    # Sempra Energy
    "T",      # AT&T Inc.
    "TMO",    # Thermo Fisher Scientific Inc.
    "TSLA",   # Tesla Inc.
    "UNH",    # UnitedHealth Group Inc.
    "USB",    # U.S. Bancorp
    "V",      # Visa Inc.
    "VZ",     # Verizon Communications Inc.
    "WBA",    # Walgreens Boots Alliance Inc.
    "WEC",    # WEC Energy Group Inc.
    "WFC",    # Wells Fargo & Co.
    "WMT",    # Walmart Inc.
    "WRB",    # W. R. Berkley Corporation
    "X",      # United States Steel Corporation
    "XEL",    # Xcel Energy Inc.
    "XYL",    # Xylem Inc.
    "ZBH",    # Zimmer Biomet Holdings Inc.
    "ZTS",    # Zoetis Inc.

    # Additional high-quality stocks
    "ABT",    # Abbott Laboratories
    "ADP",    # Automatic Data Processing Inc.
    "AFL",    # Aflac Inc.
    "ALL",    # Allstate Corporation
    "AMT",    # American Tower Corporation
    "AVGO",   # Broadcom Inc.
    "BIDU",   # Baidu Inc.
    "BK",     # Bank of New York Mellon Corp.
    "BNS",    # Bank of Nova Scotia
    "BXP",    # Boston Properties Inc.
    "CB",     # Chubb Limited
    "CHTR",   # Charter Communications Inc.
    "CL",     # Colgate-Palmolive Company
    "CLX",    # Clorox Company
    "CMCSA",  # Comcast Corporation
    "CMI",    # Cummins Inc.
    "CVS",    # CVS Health Corporation
    "DELL",   # Dell Technologies Inc.
    "DTE",    # DTE Energy Company
    "DUO",    # Duo Security Inc.
    "EMR",    # Emerson Electric Co.
    "EQR",    # Equity Residential
    "EQT",    # EQT Corporation
    "FRT",    # Federal Realty Investment Trust
    "GWW",    # Grainger (W.W.) Inc.
    "HAS",    # Hasbro Inc.
    "HLT",    # Hilton Worldwide Holdings Inc.
    "HSIC",   # Henry Schein Inc.
    "IDXX",   # IDEXX Laboratories Inc.
    "ILMN",   # Illumina Inc.
    "KMB",    # Kimberly-Clark Corporation
    "LRCX",   # Lam Research Corporation
    "LUV",    # Southwest Airlines Co.
    "MTD",    # Mettler-Toledo International Inc.
    "NEE",    # NextEra Energy Inc.
    "NRG",    # NRG Energy Inc.
    "NTRS",   # Northern Trust Corporation
    "ODFL",   # Old Dominion Freight Line Inc.
    "OMC",    # Omnicom Group Inc.
    "PCG",    # Pacific Gas and Electric Company
    "PEP",    # PepsiCo Inc.
    "RNG",    # RingCentral Inc.
    "ROP",    # Roper Technologies Inc.
    "SNA",    # Snap-on Incorporated
    "SPG",    # Simon Property Group Inc.
    "STX",    # Seagate Technology Holdings PLC
    "SYY",    # Sysco Corporation
    "TGT",    # Target Corporation
    "TTWO",   # Take-Two Interactive Software Inc.
    "UNP",    # Union Pacific Corporation
    "YUM",    # Yum! Brands Inc.

    # Meme stocks
    "AMC",    # AMC Entertainment Holdings Inc.
    "BB",     # BlackBerry Limited
    "GME",    # GameStop Corp.
    "MULN",   # Mullen Automotive Inc.
    "PLTR",   # Palantir Technologies Inc.
    "WKHS",   # Workhorse Group Inc.
    "COIN",   # Coinbase Global Inc.

    # Cyclic stocks
    "DHI",    # D.R. Horton, Inc.
    "DE",     # Deere & Company
    "ETN",    # Eaton Corporation plc
    "GM",     # General Motors Company
    "LEN",    # Lennar Corporation
    "MPC",    # Marathon Petroleum Corporation
    "NUE",    # Nucor Corporation
    "PGR",    # Progressive Corporation
    "TAP",    # Molson Coors Beverage Company
    "WHR",    # Whirlpool Corporation
    "XOM",    # Exxon Mobil Corporation
]



MERGED_TRAINING_SYMBOLS = [
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