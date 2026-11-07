# Categorize new transactions

Input: `work/new.csv` (written by `ingest.py`). Rows with a blank `category` are yours. Do **not** touch rows already filled, except to fix an obvious error.

## Rules

1. Fill `category` with exactly one key below. Never invent categories.
2. Debit = money out, credit = money in. A credit in a spending category is a refund (leave it there).
3. Credit-card payments (`PAYMENT THANK YOU`), moves between our own accounts, and e-transfers between **us** are `transfers`.
4. Pay/salary = `income`; government benefits, tax refunds, cashback = `rebates`.
5. `owner` is pre-set from the account (and the Card Member column). Change it to `Joint` only for clear shared costs on a shared account (rent, utilities, shared groceries) or when the user told you so.
6. Unsure? Pick the best guess and put `?` in `needs_review`. Do not stop to ask.
7. After filling, append each **new merchant** to `merchants.csv` (`pattern,category,owner_override`): pattern = distinctive uppercase merchant words without digits/city (e.g. `LOBLAWS`), only if that merchant always maps to one category. Skip ambiguous ones (Amazon, Walmart, e-transfers).
8. Efficient method: `python -c` / pandas to list unique blank descriptions, decide per merchant, then assign in bulk. Do not edit row by row.

## Categories (key: meaning, real examples from our history)

- `housing` (spending): Rent, mortgage, property tax, home insurance, e-transfers to landlord/roommates. e.g. Internet Banking E-TRANSFER 0107403146; Internet Banking E-TRANSFER 0107516829; Internet Banking E-TRANSFER 1046070461
- `utilities` (spending): Electricity, water, gas, internet, phone, insurance premiums. e.g. Electronic Funds Transfer PREAUTHORIZE; PROVIDENT-E-UTIL-PMNT 416-736-0630, ON; PAYMENTUS-SERVICE-FEE 800-420-1663, ON
- `transit` (spending): Public transit and micromobility: TTC/Presto, subway/bus/train, ferries, bike share, e-scooters. e.g. PRESTO FARE; BIKE SHARE TORONTO; MTA NYCT
- `automobile` (spending): Car rentals and car-share subscriptions, gas, car insurance, taxis, Uber/Lyft/Grab rideshares, parking. e.g. UBER TRIP TORONTO; SHELL; ENTERPRISE CANADA; COMMUNAUTO (owner=Joint); HERTZ; TORONTO PARKING AUTHORITY. Not Uber Eats (restaurants). Tenant insurance (Westland) is utilities.
- `groceries` (spending): Supermarkets and food shopping. e.g. IC* INSTACART HALIFAX MID-H, NS; FARM BOY #27 TORONTO, ON; IC* INSTACART*SUBSCRIP HALIFAX MID-H, 
- `restaurants` (spending): Restaurants, takeout, delivery, bars. e.g. SALAD KING TORONTO, ON; BOSTON PIZZA # 469 TORONTO, ON; TIM HORTONS #0600 BEAMSVILLE, ON
- `coffee` (spending): Coffee shops, tea. e.g.  Saltwater Purchase Cash ; TIM HORTONS #3021 TORONTO; COLUMBUS TORONTO 001 TORONTO
- `alcohol` (spending): Alcohol, cannabis. e.g. LCBO/RAO #0511 TORONTO, ON; LCBO/RAO #0360 NORTH YORK, ON; LCBO/RAO #646 TORONTO, ON
- `entertainment` (spending): Movies, concerts, events, subscriptions, leisure. e.g. GRACE OMALLEYS TORONTO, ON; THE PINT PUBLIC HOUSE TORONTO, ON; Internet Banking E-TRANSFER 0107521527
- `travel` (spending): Flights, hotels, trip expenses. e.g. HILTON GARDEN INN MONTREAL, QC; Internet Banking E-TRANSFER 0107372472; Internet Banking E-TRANSFER 0107374377
- `wellness` (spending): Gym, healthcare, pharmacy, hygiene. e.g. TORONTO WATERFRONT 10K LONDON, ON; TORONTO ISLAND - SAT A LONDON, ON; RENDEZVOUS - QUEEN ST KITCHENER, ON
- `gifts` (spending): Presents, donations. e.g. ST MICHAELS CATHEDRAL TORONTO, ON; PAPE MARKET FOODLAND #675 TORONTO, ON; SHOPAGO TORONTO, ON
- `supplies` (spending): Home improvement, electronics, household supplies. e.g. SHOPPERS DRUG MART #13 TORONTO, ON; CDN TIRE STORE #00600 TORONTO, ON; SHOPPERS DRUG MART #07 MISSISSAUGA, ON
- `cats` (spending): Anything for our cats (adopted Sept 2026): food, vet, equipment, furniture, litter. e.g. PETSMART; GLOBAL PET FOODS; LIBERTY VILLAGE ANIMAL (vet); TORONTO CAT RESCUE. Amazon purchases are `cats` only when the item is cat-related (check `amazon_orders` if the user supplied order history).
- `clothing` (spending): Clothes and shoes. e.g. SKY DRY CLEANERS TORONTO, ON; Uniqlo Canada POS_CA_YSC_ Toronto, ON; HUDSON'S BAY #1554 NORTH YORK, ON
- `purchases` (spending): Other larger or one-off purchases. e.g. SP LOOPUS HTTPSUS.LOOPE, NY; SANDBANKS ESTATE WINER WELLINGTON, ON; LIGHTHALL VINEYARDS MILFORD, ON
- `fees` (spending): Bank fees, card fees, interest charged. e.g. Electronic Funds Transfer NETWORK TRAN; Branch Transaction NETWORK FEE DISCOUN; Branch Transaction SERVICE CHARGE DISC
- `cash` (spending): ATM cash withdrawals. e.g. Automated Banking Machine ATM WITHDRAW; Internet Banking E-TRANSFER 0108314613; Internet Banking E-TRANSFER 0108311339
- `capital_gl` (spending): Investment interest, capital gains (credit) / losses (debit). e.g. CASH INTEREST; PURCHASE INTEREST; Branch Transaction INTEREST
- `savings` (spending): Contributions to savings or registered accounts treated as outflow. e.g. none yet
- `investments` (spending): Money moved into / gained from investments. e.g. none yet
- `income` (income): Employment pay and other earnings. e.g. Electronic Funds Transfer PAY 10479528; Electronic Funds Transfer PAY 10480018; Electronic Funds Transfer PAY 10480539
- `rebates` (income): Government benefits, tax refunds, cashback, reimbursements. e.g. Electronic Funds Transfer DEPOSIT CANA; CASHBACK/REMISE EN ARGENT;  CANADA Direct deposit Cash 
- `transfers` (excluded): Moves between own accounts and credit-card payments. e.g. Internet Banking INTERNET TRANSFER 000; PAYMENT THANK YOU/PAIEMEN T MERCI; Internet Banking INTERNET BILL PAY 000
