# Data dictionary

Country: United States. Period: 2000-2022 (23 annual observations; 276 monthly rows after interpolation).

| Column | Variable | Source | Indicator code / variable | Unit | Extracted |
|---|---|---|---|---|---|
| Year | Calendar year (annual file) / first day of month (monthly file) | - | - | - | - |
| CO2 | Annual production-based CO2 emissions | Our World in Data, CO2 and greenhouse-gas emissions dataset (Global Carbon Budget) | `co2` | Mt CO2 | 30 Mar 2024 |
| EA | Population aged 25+ with at least a bachelor's or equivalent degree | World Bank WDI | SE.TER.CUAT.BA.ZS | % | 20 Oct 2025 |
| GDP | GDP growth (annual, constant prices) | World Bank WDI | NY.GDP.MKTP.KD.ZG | % | 20 Oct 2025 |
| HE | Current health expenditure per capita | World Bank WDI | SH.XPD.CHEX.PC.CD | current USD | 20 Oct 2025 |
| NM | Nurses and midwives per 1,000 people | World Bank WDI | SH.MED.NUMW.P3 | per 1,000 | 20 Oct 2025 |
| UE | Unemployment, total (% of labour force, modelled ILO estimate) | World Bank WDI | SL.UEM.TOTL.ZS | % | 20 Oct 2025 |
| UP | Urban population (% of total population) | World Bank WDI | SP.URB.TOTL.IN.ZS | % | 20 Oct 2025 |
| LE | Life expectancy at birth, total (target) | World Bank WDI | SP.DYN.LE00.IN | years | 20 Oct 2025 |

`monthly_interpolated.xlsx`: the January row of each year holds the observed annual value; the other eleven months are
linear interpolations toward the next January, computed in EViews 10 (IHS Global Inc.); the months of 2022 repeat the 2022 value. No values were missing.
