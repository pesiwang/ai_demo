# html documents
from langchain.document_loaders import UnstructuredURLLoader

#urls = [
#    "https://www.understandingwar.org/backgrounder/russian-offensive-campaign-assessment-february-8-2023",
#    "https://www.understandingwar.org/backgrounder/russian-offensive-campaign-assessment-february-9-2023"
#]

urls = [
    "https://www.baidu.com",
    "https://www.google.com"
]

loader = UnstructuredURLLoader(urls=urls)

#data = loader.load()
#
#print(data)
#

# Selenium: load pages that require JavaScript to render.

from langchain.document_loaders import SeleniumURLLoader

urls = [
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://goo.gl/maps/NDSHwePEyaHMFGwh8"
]

loader = SeleniumURLLoader(urls=urls)

#data = loader.load()
#
#print(data)
#

# Playwright URL Loader : load pages that need JavaScript to render

from langchain.document_loaders import PlaywrightURLLoader

loader = PlaywrightURLLoader(urls=urls, remove_selectors=["header", "footer"])
data = loader.load()

print(data)




