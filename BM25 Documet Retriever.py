import bm25s

corpus = [

    # D1 - European Electricity Spot Market
    """The European electricity spot market allows participants
    to trade electricity through day-ahead and intraday markets.
    EPEX SPOT operates trading platforms across Europe.
    In 2025, EPEX SPOT recorded 917.5 TWh of electricity trading.
    Day-ahead auctions determine electricity prices before delivery,
    while intraday trading allows market participants to adjust
    positions closer to real-time electricity consumption.""",


    # D2 - German Power Futures Market
    """The German electricity futures market allows producers,
    utilities and energy traders to hedge price risks.
    EEX offers German Power Futures contracts for different
    delivery periods, including months, quarters and years.
    These financial instruments help companies manage
    electricity price volatility.
    The German Power Future is an important benchmark
    for European electricity derivatives trading.""",


    # D3 - European Electricity Market Integration
    """European electricity market integration aims to improve
    cross-border electricity trading and price convergence.
    ACER monitors the development of European wholesale
    electricity markets, including forward, day-ahead,
    intraday and balancing markets.
    Market coupling allows electricity to flow between
    countries according to available transmission capacity.
    Grid congestion and limited cross-border capacity
    can create electricity price differences.""",


    # D4 - Redispatch and Grid Congestion
    """Redispatch is used by German electricity network operators
    to prevent congestion in the transmission and distribution grids.
    When electricity generation creates network constraints,
    grid operators may reduce generation in one location
    and increase generation in another.
    Redispatch 2.0 includes renewable energy installations
    and requires coordination between network operators.
    Redispatch measures help maintain electricity grid security.""",


    # D5 - Electricity Balancing Markets
    """Electricity balancing markets ensure that electricity
    generation and consumption remain balanced in real time.
    European transmission system operators use balancing
    reserves to correct unexpected deviations.
    The PICASSO platform supports the exchange of automatic
    frequency restoration reserves, known as aFRR.
    The MARI platform supports manually activated frequency
    restoration reserves, known as mFRR.
    Balancing markets contribute to electricity system stability.""",


    # D6 - European Natural Gas and LNG Markets
    """The European natural gas market has changed significantly
    following the reduction of Russian pipeline gas imports.
    Liquefied natural gas, known as LNG, has become an important
    source of European gas supply.
    ACER reported that the European Union imported 146 bcm
    of LNG in 2025.
    Global LNG supply, geopolitical tensions and gas demand
    influence European wholesale natural gas prices.""",


    # D7 - Renewable Energy and Electricity Generation
    """Renewable energy sources are transforming European
    electricity generation.
    Wind and solar power represented approximately 30 percent
    of European Union electricity generation in 2025.
    Increasing renewable electricity production reduces
    dependence on fossil fuels.
    However, variable wind and solar generation creates
    additional challenges for electricity market flexibility,
    grid stability and energy storage.""",


    # D8 - Battery Storage and Energy Flexibility
    """Battery energy storage systems help balance electricity
    supply and demand.
    Batteries can store electricity when renewable generation
    is high and electricity prices are low.
    Stored electricity can be released during periods
    of high demand and higher market prices.
    Energy storage supports renewable integration,
    reduces price volatility and provides flexibility
    to European electricity markets.""",


    # D9 - European Carbon Emissions Trading
    """The European Union Emissions Trading System, known
    as EU ETS, establishes a market for carbon emission allowances.
    Electricity producers and industrial companies covered
    by the system must surrender allowances for their emissions.
    The cap-and-trade mechanism limits total emissions.
    Carbon allowance prices influence electricity generation
    costs, particularly for coal-fired and gas-fired
    power plants operating in European electricity markets.""",


    # D10 - European Electricity Market Reform
    """The European electricity market reform aims to improve
    consumer protection and reduce exposure to short-term
    electricity price volatility.
    Power Purchase Agreements, known as PPAs, allow companies
    to secure electricity through long-term contracts.
    Two-way Contracts for Difference can provide greater
    revenue stability for supported electricity generation.
    These mechanisms encourage renewable energy investment
    and improve long-term electricity price predictability."""

]

corpus_tokens=bm25s.tokenize(corpus)

queries = [
    "German electricity grid congestion",
    "European natural gas prices",
    "Renewable energy storage"
]


retriever=bm25s.BM25(k1=1.5,b=0.75)

retriever.index(corpus_tokens)

for i in range(len(queries)):
    query_tokens=bm25s.tokenize(queries[i])
    result,score=retriever.retrieve(query_tokens,corpus,k=3)
    print(result,"\n")