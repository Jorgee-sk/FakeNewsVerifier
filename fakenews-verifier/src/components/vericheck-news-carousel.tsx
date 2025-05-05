import { useState } from 'react';
import { useRef } from 'react';
import { SquareChevronDown } from 'lucide-react';
import { SquareChevronUp } from 'lucide-react';
import { ChevronLeft } from 'lucide-react';
import { ChevronRight } from 'lucide-react';
import { Link as LinkIcon } from 'lucide-react';
import { Info } from 'lucide-react';

export default function VeriCheckNewsCarousel() {
  const [currentSlide, setCurrentSlide] = useState(0);
  const [visibleNewsCount, setVisibleNewsCount] = useState(9); // Start with 9 news items
  const newsContainerRef = useRef<HTMLDivElement>(null);
  const allNews = [
    // Featured news (first 5)
    {
      title: "Nueva tecnología de energía renovable bate récords de eficiencia",
      source: "BBC",
      date: "23/04/2025",
      veracity: 95,
      image: "https://images.pexels.com/photos/20787/pexels-photo.jpg?auto=compress&cs=tinysrgb&h=350",
      url: "https://www.github.com"
    },
    {
      title: "Innovadora startup desarrolla solución para la escasez de agua",
      source: "Reuters",
      date: "22/04/2025",
      veracity: 75,
      image: "https://i.pinimg.com/736x/0e/21/55/0e2155832173482a1b4a71f9565b68f1.jpg",
      url: "https://www.github.com"
    },
    {
      title: "España ha adjudicado 46 contratos por más de 1.000 millones a la industria militar israelí desde el inicio de la guerra de Gaza",
      source: "El País",
      date: "21/04/2025",
      veracity: 90,
      image: "https://imagenes.elpais.com/resizer/v2/4AJ4QIH5TG6KYXP2MJSLQBJTNA.jpg?auth=7ab4cbd4c743b5c7927af0effd5e4c4025fb23598d963d9328c9880ddad40398&width=1200&height=675&smart=true",
      url: "https://www.github.com"
    },
    {
      title: "Nuevo avance en inteligencia artificial para procesamiento de lenguaje",
      source: "Tech Today",
      date: "20/04/2025",
      veracity: 85,
      image: "https://i.pinimg.com/236x/d9/5d/d6/d95dd6ec52ad53237b4c42258f043a0a.jpg",
      url: "https://www.github.com"
    },
    {
      title: "Descubren especie marina en aguas profundas del Pacífico",
      source: "National Geographic",
      date: "19/04/2025",
      veracity: 98,
      image: "https://i.pinimg.com/236x/7a/bc/9a/7abc9a265cce49c2101d59165d632c54.jpg",
      url: "https://www.github.com"
    },
    {
      title: "El cambio climático acelera la pérdida de biodiversidad en zonas tropicales",
      source: "Science Daily",
      date: "18/04/2025",
      veracity: 92,
      image: "https://i.pinimg.com/236x/02/2a/10/022a10b80bf98bc052273961decc207d.jpg",
      url: "https://www.github.com"
    },
    {
      title: "Nueva política energética provoca debate entre expertos del sector",
      source: "Financial Times",
      date: "17/04/2025",
      veracity: 67,
      image: "https://i.pinimg.com/236x/5b/c7/d1/5bc7d1e2ceb2bc26676ac37814436ac4.jpg",
      url: "https://www.github.com"
    },
    {
      title: "Aumento de casos de ciberataques preocupa a autoridades internacionales",
      source: "The Guardian",
      date: "16/04/2025",
      veracity: 88,
      image: "https://i.pinimg.com/236x/de/df/79/dedf79e4ab0decf363205c15b2aeda68.jpg",
      url: "https://www.github.com"
    },
    {
      title: "Estudio vincula consumo excesivo de redes sociales con problemas de salud mental",
      source: "Psychology Today",
      date: "15/04/2025",
      veracity: 77,
      image: "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ2ZGL6vQo34QtzWMInEMJQSbfJiNI84O1P2Q&s",
      url: "https://www.github.com"
    },
    {
      title: "Avances en fusión nuclear podrían revolucionar la producción energética",
      source: "MIT Technology Review",
      date: "14/04/2025",
      veracity: 82,
      image: "https://i.pinimg.com/236x/b2/60/94/b26094970505bcd59c2e5fe8b6f41cf0.jpg",
      url: "https://www.github.com"
    },
    {
      title: "Crisis económica afecta severamente a mercados emergentes",
      source: "Bloomberg",
      date: "13/04/2025",
      veracity: 95,
      image: "https://i.pinimg.com/236x/a5/05/f0/a505f04972bc77275a6057203fe26728.jpg",
      url: "https://www.github.com"
    },
    {
      title: "Controversia por nueva regulación en el sector tecnológico",
      source: "Wall Street Journal",
      date: "12/04/2025",
      veracity: 45,
      image: "https://ichef.bbci.co.uk/ace/standard/1024/cpsprodpb/beea/live/b7cc92f0-21da-11f0-8c2e-77498b1ce297.jpg",
      url: "https://www.github.com"
    },
    {
      title: "Descubren posibles señales de vida microbiana en otro planeta",
      source: "NASA",
      date: "11/04/2025",
      veracity: 62,
      image: "https://i.pinimg.com/236x/d6/c5/1e/d6c51edfc77775e112d90315cbcfebfa.jpg",
      url: "https://www.github.com"
    },
    {
      title: "Expertos advierten sobre nueva variante de virus estacional",
      source: "WHO",
      date: "10/04/2025",
      veracity: 89,
      image: "https://i.pinimg.com/736x/b0/99/1b/b0991be50d704dbcc7f78879f8a202db.jpg",
      url: "https://www.github.com"
    },
    {
      title: "Manifestaciones masivas en contra de nueva ley de privacidad",
      source: "Reuters",
      date: "09/04/2025",
      veracity: 81,
      image: "https://i.pinimg.com/474x/14/20/fd/1420fdb2c1b84a55bc9a61e3050b0fa5.jpg",
      url: "https://www.github.com"
    }
  ];
  
  // Filter featured news for the carousel
  const featuredNews = allNews.slice(0,5);
  // Filter non-featured news for the "All News" section
  const regularNews = allNews;
  
  // Function to get veracity color based on percentage
  const getVeracityColor = (percentage: number) => {
    if (percentage >= 75) return "bg-green-500";
    if (percentage >= 50) return "bg-yellow-500";
    return "bg-red-500";
  };
  
  const nextSlide = () => {
    setCurrentSlide((prev) => (prev === featuredNews.length - 1 ? 0 : prev + 1));
  };
  
  const prevSlide = () => {
    setCurrentSlide((prev) => (prev === 0 ? featuredNews.length - 1 : prev - 1));
  };
  
  const goToSlide = (index: number) => {
    setCurrentSlide(index);
  };
  
  // Function to get correct index with wrapping
  const getSlideIndex = (baseIndex: number, offset: number) => {
    const totalSlides = featuredNews.length;
    return (baseIndex + offset + totalSlides) % totalSlides;
  };
  
  // Get indices for the three visible slides
  const leftIndex = getSlideIndex(currentSlide, -1);
  const centerIndex = currentSlide;
  const rightIndex = getSlideIndex(currentSlide, 1);
  
  // Function to show more news
  const showMoreNews = () => {
    // Add 3 more news items (1 row) at a time
    const next = setVisibleNewsCount(prev => Math.min(prev + 3, regularNews.length));
    setTimeout(() => {
      if (newsContainerRef.current) {
        newsContainerRef.current.scrollIntoView({ behavior: 'smooth', block: 'end' });
      }
    }, 100); // pequeño delay para esperar a que React renderice
    return next;
  };
  
  // Function to show less news
  const showLessNews = () => {
    // Remove 3 news items (1 row) at a time, but keep at least 9
    setVisibleNewsCount(prev => Math.max(prev - 3, 9));
  };
  
  // Create rows of news items
  type NewsItem = {
    title: string;
    source: string;
    date: string;
    veracity: number;
    image: string;
    url: string;
  };
  
  const newsRows: (NewsItem | null)[][] = [];

  for (let i = 0; i < Math.min(visibleNewsCount, regularNews.length); i += 3) {
    const rowItems = regularNews.slice(i, i + 3) as (NewsItem | null)[];
    if (rowItems.length === 3) {
      newsRows.push(rowItems);
    } else {
      const paddedRow = [...rowItems];
      while (paddedRow.length < 3) {
        paddedRow.push(null);
      }
      newsRows.push(paddedRow);
    }
  }
  
  return (
    <div className="min-h-screen bg-gradient-to-br from-white via-blue-50 to-blue-200 w-full flex flex-col items-center">
      <div className="w-full max-w-4xl px-4">
        <h2 className="text-black text-2xl font-bold text-center my-4">Noticias Destacadas</h2>
        
        <div className="relative flex justify-center mb-4">
          {/* Left arrow outside carousel */}
          <button 
            onClick={prevSlide}
            className="absolute left-0 top-1/2 transform -translate-y-1/2 -translate-x-10 bg-black text-white rounded-full w-8 h-8 flex items-center justify-center z-10"
          >
           <ChevronLeft className="w-4 h-4"/>
          </button>
          
          <div className="w-full">
            <div className="flex relative">
              {/* Left (smaller) slide - vertically centered */}
              <div className="w-1/4 px-2 flex items-center">
                <div className="bg-gray-100 rounded-lg overflow-hidden shadow">
                  <img src={featuredNews[leftIndex].image} alt={featuredNews[leftIndex].title} className="w-full h-32 object-cover" />
                  <div className="p-2">
                    <h3 className="text-xs font-semibold truncate">{featuredNews[leftIndex].title}</h3>
                    <div className="flex justify-between text-xs mt-1">
                      <p className='truncate'>Fuente: {featuredNews[leftIndex].source}</p>
                      <p>{featuredNews[leftIndex].date}</p>
                    </div>
                    <div className="mt-1 bg-gray-200 rounded-full h-1">
                      <div 
                        className={`${getVeracityColor(featuredNews[leftIndex].veracity)} h-1 rounded-full`} 
                        style={{ width: `${featuredNews[leftIndex].veracity}%` }}
                      ></div>
                    </div>
                    <p className="text-xs text-right flex items-center justify-end mt-2">
                            <button onClick={() => console.log('Mostrar información')} className="focus:outline-none mr-1 flex items-center">
                              <Info className="w-3 h-3 hover:text-blue-500 cursor-pointer" />
                            </button>
                            <span>Veracidad: {featuredNews[leftIndex].veracity}%</span>
                    </p>
                  </div>
                </div>
              </div>
              
              {/* Center (main) slide */}
              <div className="w-2/4 px-2">
                <div className="bg-gray-100 rounded-lg overflow-hidden shadow">
                  <img src={featuredNews[centerIndex].image} alt={featuredNews[centerIndex].title} className="w-full h-48 object-cover" />
                  <div className="p-3">
                    <h3 className="text-lg font-semibold ">{featuredNews[centerIndex].title}<a href={featuredNews[centerIndex].url}>
                    <LinkIcon className="w-4 h-4 mr-1 hover:text-blue-500"/></a></h3>
                    <div className="flex justify-between text-sm mt-2">
                      <p>Fuente: {featuredNews[centerIndex].source}</p>
                      <p>{featuredNews[centerIndex].date}</p>
                    </div>
                    <div className="mt-2 bg-gray-200 rounded-full h-2">
                      <div 
                        className={`${getVeracityColor(featuredNews[centerIndex].veracity)} h-2 rounded-full`} 
                        style={{ width: `${featuredNews[centerIndex].veracity}%` }}
                      ></div>
                    </div>
                    <p className="text-sm text-right flex items-center justify-end mt-2">
                            <button onClick={() => console.log('Mostrar información')} className="focus:outline-none mr-1 flex items-center">
                              <Info className="w-4 h-4 hover:text-blue-500 cursor-pointer" />
                            </button>
                            <span>Veracidad: {featuredNews[centerIndex].veracity}%</span>
                          </p>
                  </div>
                </div>
              </div>
              
              {/* Right (smaller) slide - vertically centered */}
              <div className="w-1/4 px-2 flex items-center">
                <div className="bg-gray-100 rounded-lg overflow-hidden shadow">
                  <img src={featuredNews[rightIndex].image} alt={featuredNews[rightIndex].title} className="w-full h-32 object-cover" />
                  <div className="p-2">
                    <h3 className="text-xs font-semibold truncate">{featuredNews[rightIndex].title}</h3>
                    <div className="flex justify-between text-xs mt-1">
                      <p className='truncate'>Fuente: {featuredNews[rightIndex].source}</p>
                      <p>{featuredNews[rightIndex].date}</p>
                    </div>
                    <div className="mt-1 bg-gray-200 rounded-full h-1">
                      <div 
                        className={`${getVeracityColor(featuredNews[rightIndex].veracity)} h-1 rounded-full`} 
                        style={{ width: `${featuredNews[rightIndex].veracity}%` }}
                      ></div>
                    </div>
                    <p className="text-xs text-right flex items-center justify-end mt-2">
                            <button onClick={() => console.log('Mostrar información')} className="focus:outline-none mr-1 flex items-center">
                              <Info className="w-3 h-3 hover:text-blue-500 cursor-pointer" />
                            </button>
                            <span>Veracidad: {featuredNews[rightIndex].veracity}%</span>
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
          
          {/* Right arrow outside carousel */}
          <button 
            onClick={nextSlide}
            className="absolute right-0 top-1/2 transform -translate-y-1/2 translate-x-10 bg-black text-white rounded-full w-8 h-8 flex items-center justify-center z-10"
          >
            <ChevronRight className="w-4 h-4"/>
          </button>
        </div>
        
        <div className="flex justify-center gap-2 mb-4">
          {featuredNews.map((_, index) => (
            <button 
              key={index} 
              onClick={() => goToSlide(index)}
              className={`h-2 rounded-full ${currentSlide === index ? 'w-4 bg-blue-500' : 'w-2 bg-gray-300'}`}
            ></button>
          ))}
        </div>
        
        <div className="w-full flex justify-center mb-8">
          <div className="w-full max-w-4xl h-px bg-black"></div>
        </div>
        
        {/* All News Section - Shows exactly 9 news items initially, adds 3 at a time */}
        <div className="w-full mb-8" ref={newsContainerRef}>
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-black text-xl font-bold">Todas las Noticias</h2>
          </div>
          
          <div className="news-grid space-y-6">
            {newsRows.map((row, rowIndex) => (
              <div key={rowIndex} className="flex -mx-2">
                {row.map((news, colIndex) => (
                  news ? (
                    <div key={colIndex} className="w-1/3 px-2">
                      <div className="bg-gray-100 rounded-lg overflow-hidden shadow h-full">
                        <img src={news.image} alt={news.title} className="w-full h-32 object-cover" />
                        <div className="p-3">
                          <div className="flex justify-between items-center">
                          <h3 className="text-sm font-semibold line-clamp-2">
                            {news.title}
                          </h3>
                          <a href={news.url} className="hover:text-blue-500">
                            <LinkIcon className="w-4 h-4 mr-1"/>
                          </a>
                        </div>                          
                          <div className="flex justify-between text-xs mt-2">
                            <p>Fuente: {news.source}</p>
                            <p>{news.date}</p>
                          </div>
                          <div className="mt-2 bg-gray-200 rounded-full h-1">
                            <div 
                              className={`${getVeracityColor(news.veracity)} h-1 rounded-full`} 
                              style={{ width: `${news.veracity}%` }}
                            ></div>
                          </div>
                          <p className="text-xs text-right flex items-center justify-end mt-2">
                            <button onClick={() => console.log('Mostrar información')} className="focus:outline-none mr-1 flex items-center">
                              <Info className="w-4 h-4 hover:text-blue-500 cursor-pointer" />
                            </button>
                            <span>Veracidad: {news.veracity}%</span>
                          </p>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div key={colIndex} className="w-1/3 px-2"></div>
                  )
                ))}
              </div>
            ))}
          </div>

          <div className="flex flex-col sm:flex-row justify-center items-center mt-6 space-y-3 sm:space-y-0 sm:space-x-4">
            {visibleNewsCount < regularNews.length && (
              <button 
                onClick={showMoreNews}
                className="bg-black hover:bg-zinc-900 focus:ring-2 rounded-lg focus:ring-gray-400 transition duration-200 text-white rounded px-6 py-2 text-sm h-10 min-w-[160px] flex items-center justify-center gap-2">
                <SquareChevronDown className="w-4 h-4" /> Ver más noticias
              </button>
            )}

            {visibleNewsCount > 9 && (
              <button 
                onClick={showLessNews}
                className="bg-black hover:bg-zinc-900 focus:ring-2 rounded-lg focus:ring-gray-400 transition duration-200 text-white rounded px-6 py-2 text-sm h-10 min-w-[160px] flex items-center justify-center gap-2">
                <SquareChevronUp className="w-4 h-4" /> Ver menos
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
