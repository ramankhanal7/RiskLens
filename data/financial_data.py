import finnhub

CURR_TIMESTAMP = 7671023

#MIN_ID is not inclusive

finnhub_client = finnhub.Client(api_key="d6uaqgpr01qp1k9bsii0d6uaqgpr01qp1k9bsiig")
current_news = finnhub_client.general_news('general', min_id = CURR_TIMESTAMP)

list_of_ids = [news['id'] for news in current_news]
list_of_ids.sort()

list_of_timestamps = [(news['datetime'], news['url']) for news in current_news]
list_of_timestamps.sort(reverse=True)

print(list_of_timestamps[0][1])
print(list_of_timestamps[0][0])
print()
print(list_of_ids)