from requests import get
from bs4 import BeautifulSoup
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait
from threading import Lock
import json


class IMDbCrawler:
    """
    put your own user agent in the headers
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
    }
    top_250_URL = 'https://www.imdb.com/chart/top/'

    def __init__(self, crawling_threshold=1000):
        """
        Initialize the crawler

        Parameters
        ----------
        crawling_threshold: int
            The number of pages to crawl
        """
        self.crawling_threshold = crawling_threshold
        self.not_crawled = deque()
        self.crawled = []
        self.added_ids = set()
        self.add_list_lock = Lock()
        self.add_queue_lock = Lock()

    def get_id_from_URL(self, URL):
        """
        Get the id from the URL of the site. The id is what comes exactly after title.
        for example the id for the movie https://www.imdb.com/title/tt0111161/?ref_=chttp_t_1 is tt0111161.

        Parameters
        ----------
        URL: str
            The URL of the site
        Returns
        ----------
        str
            The id of the site
        """
        parts = URL.split('/')
        return parts[-2]

    def write_to_file_as_json(self):
        """
        Save the crawled files into json
        """
        not_crawled_list = list(self.not_crawled)

        with open('../IMDB_crawled.json', 'w') as f:
            json.dump(self.crawled, f)

        with open('../IMDB_not_crawled.json', 'w') as f:
            json.dump(not_crawled_list, f)

    def read_from_file_as_json(self):
        """
        Read the crawled files from json
        """

        with open('../IMDB_crawled.json', 'r') as f:
            data = json.load(f)
            self.crawled = data

        with open('../IMDB_not_crawled.json', 'r') as f:
            data = json.load(f)
            self.not_crawled = data

    def crawl(self, URL):
        """
        Make a get request to the URL and return the response

        Parameters
        ----------
        URL: str
            The URL of the site
        Returns
        ----------
        requests.models.Response
            The response of the get request
        """
        response = get(URL, headers=self.headers)
        return response

    def extract_top_250(self):
        """
        Extract the top 250 movies from the top 250 page and use them as a seed for the crawler to start crawling.
        """
        response = self.crawl(self.top_250_URL)

        if response.status_code == 200:
          soup = BeautifulSoup(response.text, 'html.parser')
          movie_links = soup.findAll('a', {'class': "ipc-title-link-wrapper"})
          for link in movie_links:
            if link['href'].startswith('/title'):
              movie_url = 'https://www.imdb.com' + link['href']
              movie_id = self.get_id_from_URL(movie_url)

              if movie_id not in self.added_ids and movie_url not in self.crawled:
                self.not_crawled.append(movie_url)
                self.added_ids.add(movie_id)

    def get_imdb_instance(self):
        return {
            'id': None,  # str
            'title': None,  # str
            'first_page_summary': None,  # str
            'release_year': None,  # str
            'mpaa': None,  # str
            'budget': None,  # str
            'gross_worldwide': None,  # str
            'rating': None,  # str
            'directors': None,  # List[str]
            'writers': None,  # List[str]
            'stars': None,  # List[str]
            'related_links': None,  # List[str]
            'genres': None,  # List[str]
            'languages': None,  # List[str]
            'countries_of_origin': None,  # List[str]
            'summaries': None,  # List[str]
            'synopsis': None,  # List[str]
            'reviews': None,  # List[[str]]
        }

    def start_crawling(self):
        """
        Start crawling the movies until the crawling threshold is reached.

        ThreadPoolExecutor is used to make the crawler faster by using multiple threads to crawl the pages.
        You are free to use it or not. If used, not to forget safe access to the shared resources.
        """
        self.extract_top_250()
        futures = []
        crawled_counter = 0

        with ThreadPoolExecutor(max_workers=20) as executor:
            while crawled_counter < self.crawling_threshold and self.not_crawled:
                with self.add_queue_lock:
                    URL = self.not_crawled.popleft()
                futures.append(executor.submit(self.crawl_page_info, URL))
                crawled_counter += 1
                if len(self.not_crawled) == 0:
                    wait(futures)
                    futures = []
            wait(futures)

        # while len(self.crawled) < self.crawling_threshold and self.not_crawled:
          # URL = self.not_crawled.popleft()
        # futures.append(executor.submit(self.crawl_page_info, URL))
          # self.crawl_page_info(URL)


    def crawl_page_info(self, URL):
        """
        Main Logic of the crawler. It crawls the page and extracts the information of the movie.
        Use related links of a movie to crawl more movies.

        Parameters
        ----------
        URL: str
            The URL of the site
        """
        print("Crawling:", URL)

        response = self.crawl(URL)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            movie_info = self.get_imdb_instance()
            self.extract_movie_info(soup, movie_info, URL)
            with self.add_queue_lock:
                self.crawled.append(movie_info)
                print(len(self.crawled))

            related_links = movie_info['related_links']
            for link in related_links:
                if link not in self.not_crawled:
                    with self.add_queue_lock:
                        self.not_crawled.append(link)
                    with self.add_list_lock:
                        self.added_ids.add(self.get_id_from_URL(link))

    def extract_movie_info(self, res, movie, URL):
        """
        Extract the information of the movie from the response and save it in the movie instance.

        Parameters
        ----------
        res: requests.models.Response
            The response of the get request
        movie: dict
            The instance of the movie
        URL: str
            The URL of the site
        """
        movie['id'] = self.get_id_from_URL(URL)
        movie['title'] = self.get_title(res)
        movie['first_page_summary'] = self.get_first_page_summary(res)
        movie['release_year'] = self.get_release_year(res)
        movie['mpaa'] = self.get_mpaa(URL)
        movie['summaries'] = self.get_summary(URL)
        movie['budget'] = self.get_budget(res)
        movie['gross_worldwide'] = self.get_gross_worldwide(res)
        movie['directors'] = self.get_director(res)
        movie['writers'] = self.get_writers(res)
        movie['stars'] = self.get_stars(res)
        movie['related_links'] = self.get_related_links(res)
        movie['genres'] = self.get_genres(res)
        movie['languages'] = self.get_languages(res)
        movie['countries_of_origin'] = self.get_countries_of_origin(res)
        movie['rating'] = self.get_rating(res)
        movie['synopsis'] = self.get_synopsis(URL)
        movie['reviews'] = self.get_reviews_with_scores(URL)


    def get_summary_link(self, url):
        """
        Get the link to the summary page of the movie
        Example:
        https://www.imdb.com/title/tt0111161/ is the page
        https://www.imdb.com/title/tt0111161/plotsummary is the summary page

        Parameters
        ----------
        url: str
            The URL of the site
        Returns
        ----------
        str
            The URL of the summary page
        """
        try:
            movie_id = url.split('/')[-2]
            summary_link = f'https://www.imdb.com/title/{movie_id}/plotsummary'
            return summary_link
            pass
        except:
            print("failed to get summary link")

    def get_review_link(self, url):
        """
        Get the link to the review page of the movie
        Example:
        https://www.imdb.com/title/tt0111161/ is the page
        https://www.imdb.com/title/tt0111161/reviews is the review page
        """
        try:
            movie_id = url.split('/')[-2]
            summary_link = f'https://www.imdb.com/title/{movie_id}/reviews'
            return summary_link
        except:
            print("failed to get review link")

    def get_title(self, soup):
        """
        Get the title of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The title of the movie

        """
        try:
            title_tag = soup.find('h1')
            if title_tag:
                return title_tag.text.strip()
        except:
            print("failed to get title")
            return ''


    def get_first_page_summary(self, soup):
        """
        Get the first page summary of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The first page summary of the movie
        """
        try:
            summary = soup.find('span', {'class': 'sc-466bb6c-0 hlbAws'})
            return summary.text.strip()
        except:
            print("failed to get first page summary")
            return ''

    def get_director(self, soup):
        """
        Get the directors of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The directors of the movie
        """
        try:
            director_tag = soup.find('script', {'id': '__NEXT_DATA__', "type": "application/json"})
            data = json.loads(director_tag.contents[0])
            credits_info = data['props']['pageProps']['mainColumnData']['directors'][0]['credits']
            directors = [str(credit['name']['nameText']['text']) for credit in credits_info]
            if directors is not None:
                return directors
            return []
        except:
            print("failed to get director")
            return []

    def get_stars(self, soup):
        """
        Get the stars of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The stars of the movie
        """
        try:
            stars_tag = soup.find('script', {"type": "application/ld+json"})
            data = json.loads(stars_tag.contents[0])
            stars_info = data['actor']
            stars = [star['name'] for star in stars_info]
            if stars is not None:
                return stars
            return []
        except:
            print("failed to get stars")
            return ['']

    def get_writers(self, soup):
        """
        Get the writers of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The writers of the movie
        """
        try:
            writer_tag = soup.find('script', {'id': '__NEXT_DATA__', "type": "application/json"})
            data = json.loads(writer_tag.contents[0])
            credits_info = data['props']['pageProps']['mainColumnData']['writers'][0]['credits']
            writers = [credit['name']['nameText']['text'] for credit in credits_info]
            if writers is not None:
                return writers
            return []
        except:
            print("failed to get writers")
            return []

    def get_related_links(self, soup):
        """
        Get the related links of the movie from the More like this section of the page from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The related links of the movie
        """
        try:
            related_links =[]
            links = soup.findAll('a', {'class':"ipc-poster-card__title ipc-poster-card__title--clamp-2 ipc-poster-card__title--clickable"})
            for link in links:
                url = "https://www.imdb.com/" + link["href"]
                related_links.append(url)
            return related_links
        except:
            print("failed to get related links")
            return []

    def get_summary(self, URL):
        """
        Get the summary of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The summary of the movie
        """
        try:
            NewResponse = self.crawl(self.get_summary_link(URL))
            if NewResponse.status_code == 200:
                summarySoup = BeautifulSoup(NewResponse.text, 'html.parser')
                sums_tag = summarySoup.find('script', {'id': '__NEXT_DATA__', "type": "application/json"})
                data = json.loads(sums_tag.contents[0])
                sums_info = data['props']['pageProps']['contentData']['categories'][0]['section']['items']
                summaries = [summary['htmlContent'] for summary in sums_info]
                if summaries:
                    return summaries
                return []
        except:
            print("failed to get summary")
            return []

    def get_synopsis(self, URL):
        """
        Get the synopsis of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The synopsis of the movie
        """
        try:
            NewResponse = self.crawl(self.get_summary_link(URL))
            if NewResponse.status_code == 200:
                summarySoup = BeautifulSoup(NewResponse.text, 'html.parser')
                synopsis_tag = summarySoup.find('script', {'id': '__NEXT_DATA__', "type": "application/json"})
                data = json.loads(synopsis_tag.contents[0])
                synopsis = data['props']['pageProps']['contentData']['categories'][1]['section']['items'][0]['htmlContent']
                if synopsis is not None:
                    return [synopsis]
                return []
        except:
            print("failed to get synopsis")
            return []

    def get_reviews_with_scores(self, URL):
        """
        Get the reviews of the movie from the soup
        reviews structure: [[review,score]]

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[List[str]]
            The reviews of the movie
        """
        try:
            NewResponse = self.crawl(self.get_review_link(URL))
            if NewResponse.status_code == 200:
                summarySoup = BeautifulSoup(NewResponse.text, 'html.parser')
                review_divs = summarySoup.select('div[class^="lister-item mode-detail imdb-user-review"]')
                reviews = []
                for div in review_divs:
                    span = div.find('span', class_='point-scale')
                    review = div.find('div', class_='text show-more__control').text
                    if span:
                        reviews.append((review, span.previous_sibling.text))
                    else:
                        reviews.append((review, 'No rating'))
                return reviews
        except:
            print("failed to get reviews")
            return []

    def get_genres(self, soup):
        """
        Get the genres of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The genres of the movie
        """
        try:
            genres_tag = soup.find('script', {'id': '__NEXT_DATA__', "type": "application/json"})
            data = json.loads(genres_tag.contents[0])
            AllGenres = data['props']['pageProps']['aboveTheFoldData']['genres']['genres']
            genres = [genre['text'] for genre in AllGenres]
            if genres:
                return genres
            return []
        except:
            print("Failed to get generes")
            return []

    def get_rating(self, soup):
        """
        Get the rating of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The rating of the movie
        """
        try:
            rating_tag = soup.find('script', {"type": "application/ld+json"})
            data = json.loads(rating_tag.contents[0])
            rating = data['aggregateRating']['ratingValue']
            return str(rating)
        except:
            print("failed to get rating")
            return ''

    def get_mpaa(self, URL):
        """
        Get the MPAA of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The MPAA of the movie
        """
        try:
            newURL = f'https://www.imdb.com/title/{self.get_id_from_URL(URL)}/parentalguide'
            NewResponse = self.crawl(newURL)
            if NewResponse.status_code == 200:
                mpaaSoup = BeautifulSoup(NewResponse.text, 'html.parser')
                mpaa_row = mpaaSoup.find('tr', {'id': 'mpaa-rating'})
                mpaa_text = mpaa_row.find('td', {'class': 'ipl-zebra-list__label'}).find_next('td').text.strip()
                if mpaa_text:
                    return mpaa_text
                return ''
        except:
            print("failed to get mpaa")
            return ''

    def get_release_year(self, soup):
        """
        Get the release year of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The release year of the movie
        """
        try:
            release_year_tag = soup.find('script', {'id': '__NEXT_DATA__'})
            data = json.loads(release_year_tag.contents[0])
            release_year = data['props']['pageProps']['aboveTheFoldData']['releaseYear']['year']
            return str(release_year)
        except:
            print("failed to get release year")
            return ''

    def get_languages(self, soup):
        """
        Get the languages of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The languages of the movie
        """
        try:
            language_tag = soup.find('script', {'id': '__NEXT_DATA__', "type": "application/json"})
            data = json.loads(language_tag.contents[0])
            language_info = data['props']['pageProps']['mainColumnData']['spokenLanguages']['spokenLanguages']
            languages = [language['text'] for language in language_info]
            return languages

        except:
            print("failed to get languages")
            return []

    def get_countries_of_origin(self, soup):
        """
        Get the countries of origin of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The countries of origin of the movie
        """
        try:
            country_tag = soup.find('script', {'id': '__NEXT_DATA__', "type": "application/json"})
            data = json.loads(country_tag.contents[0])
            country_info = data['props']['pageProps']['mainColumnData']['countriesOfOrigin']['countries']
            countries = [country['text'] for country in country_info]
            return countries
        except:
            print("failed to get countries of origin")
            return []

    def get_budget(self, soup):
        """
        Get the budget of the movie from box office section of the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The budget of the movie
        """
        try:
            budget_tag = soup.find('script', {'id': '__NEXT_DATA__', "type":"application/json"})
            data = json.loads(budget_tag.contents[0])
            budget = data['props']['pageProps']['mainColumnData']['productionBudget']['budget']['amount']
            return str(budget)

        except:
            print("failed to get budget")
            return ''

    def get_gross_worldwide(self, soup):
        """
        Get the gross worldwide of the movie from box office section of the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The gross worldwide of the movie
        """
        try:
            gross_tag = soup.find('script', {'id': '__NEXT_DATA__', "type": "application/json"})
            data = json.loads(gross_tag.contents[0])
            gross_worldwide = data['props']['pageProps']['mainColumnData']['worldwideGross']['total']['amount']
            if gross_worldwide is not None:
              return str(gross_worldwide)
            return ''

        except:
            print("failed to get gross worldwide")
            return ''


def main():
    imdb_crawler = IMDbCrawler(crawling_threshold=1000)
    # imdb_crawler.read_from_file_as_json()
    imdb_crawler.start_crawling()
    imdb_crawler.write_to_file_as_json()


if __name__ == '__main__':
    main()

