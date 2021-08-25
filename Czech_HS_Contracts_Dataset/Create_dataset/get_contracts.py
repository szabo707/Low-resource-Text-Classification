import jsonlines
import json
import csv
import time

from urllib.request import Request, urlopen
from itertools import islice

#OBLAST = 'it_sw'
#NAZOV = '10002_IT_Software'

NUMBER_OF_CONTRACTS = 1000

headers = {
            'Accept': 'application/json',
            'Authorization': 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx' #unique authentication token for authorization
        }

with open('notations_of_contracts', 'r') as file:
    csv_reader = csv.reader(file, delimiter=' ')
    for row in islice(csv_reader,0,105):
        OBLAST = str(row[2])
        NAZOV = str(row[0] + '_' + row[1])

        for strana in range(0, 42):
            strana = strana + 1
            url = 'https://www.hlidacstatu.cz/api/v2/smlouvy/hledat?dotaz=oblast%3A' + OBLAST + '&strana=' + str(strana) + '&razeni=0'
            request_hledat = Request(url, headers=headers)
            response_body_hledat = urlopen(request_hledat).read()

            data_hledat = json.loads(response_body_hledat)
            node_hledat = data_hledat.get('Results', {})

            limit = 0
            for i in node_hledat:
                contract_identifikator = i['identifikator']
                ID = contract_identifikator.get('idVerze', {})
                print(ID)
                url = 'https://www.hlidacstatu.cz/api/v2/smlouvy/' + str(ID)
                print(url)

                request = Request(url, headers=headers)
                response_body = urlopen(request).read()
                time.sleep(1)

                data = json.loads(response_body)
                node = data.get('Classification', {}).get('Types')

                if node is None or len(node) == 0:
                    continue

                with jsonlines.open('contracts_by_relevance/' + NAZOV + '.jsonl', mode='a') as writer:
                    with open('categoriesCZ.json') as reader:
                        dataCategories = json.load(reader)
                        for i in node:
                            typeValue = i['TypeValue']
                            for p in dataCategories['categories']:
                                nameStr = p.get(str(typeValue))
                                if not nameStr is None:
                                    i["TypeName"] = nameStr
                        print(data)
                        writer.write(data)
                limit += 1
                if limit == NUMBER_OF_CONTRACTS:
                    break
