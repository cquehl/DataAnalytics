# Project Notes for Claude

## Output Directory Pattern
Scripts should always create the `outputs/` directory before saving files:
```python
os.makedirs('outputs', exist_ok=True)
```
This prevents `FileNotFoundError` on fresh clones. See `efa_analysis.py` for the reference implementation.

## Data Schema (Decent_Data_Set_Feb3.csv)

### Dependent Variables (DVs)
- **PRICETIERKEY** - Price tier assigned to purchaser (GUID). Max 12 tiers, numerically ordered. Drives pricing strategy and MARGINSYSTEMRECOMMENDED.
- **ISWON** - Whether quote was won (separate prediction problem)

### Independent Variables (IVs)

**Identifiers:**
- DATEESTIMATECREATED - Estimate creation date
- LINEITEMKEY - Specific line item in estimate
- ESTIMATEKEY - Groups line items into single estimate

**Price/Weight Cluster (highly correlated):**
- LINEITEMUNITPRICE, LINEITEMTOTALPRICE, ESTIMATETOTALPRICE
- ESTIMATETOTALQUANTITY, QTYPERLINEITEM
- MATERIALCENTWEIGHTCOST, MATERIALWEIGHTUNITOFMEASURE
- MATERIALUNITWEIGHT, MATERIALLINETOTALWEIGHT, MATERIALLINETOTALCENTWEIGHTCOST

**Purchaser Hierarchy:**
- PURCHASERLOCATIONKEY / PURCHASERLOCATIONNAME → PURCHASERCOMPANYNAME (umbrella)

**Supplier Hierarchy:**
- SUPPLIERLOCATIONKEY / SUPPLIERLOCATIONNAME → SUPPLIERCOMPANYNAME (umbrella)

**Product Info:**
- MATERIALTYPE, SHAPEGROUP, SHAPEDETAIL

**Other Key Variables:**
- LEADTIME - Quote lead time (stored as Excel date serial, format as date)
- MARGINSYSTEMRECOMMENDED - Margin the pricing engine recommended
- MARGINENGINETYPE - "basket" or "line item" (price breaks grouped by Material Type + Shape Group)

### Analysis Goals
1. **Predict Price Tier** - Use historical data to determine customer tier for suppliers who don't have them segregated
2. **Predict Win/Loss** - Understand what drives ISWON (separate problem)
