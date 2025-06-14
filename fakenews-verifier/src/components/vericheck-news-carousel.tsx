import { useState } from 'react';
import { useRef } from 'react';
import { SquareChevronDown } from 'lucide-react';
import { SquareChevronUp } from 'lucide-react';
import { ChevronLeft } from 'lucide-react';
import { ChevronRight } from 'lucide-react';
import { Link as LinkIcon } from 'lucide-react';
import { Info } from 'lucide-react';
import { Moon } from 'lucide-react';
import { Sun } from 'lucide-react';
import noticias from './noticias.json';

type NewsItem = {
  title: string;
  source: string;
  date: string;
  veracity: number;
  image: string;
  url: string;
  explanation?: string[];
};

export default function VeriCheckNewsCarousel() {
  const [currentSlide, setCurrentSlide] = useState(0);
  const [visibleNewsCount, setVisibleNewsCount] = useState(9);
  const [showModal, setShowModal] = useState(false);
  const [selectedNews, setSelectedNews] = useState<NewsItem | null>(null);
  const [isDarkMode, setIsDarkMode] = useState(false);
  const newsContainerRef = useRef<HTMLDivElement>(null);

  const allNews: NewsItem[] = noticias;
  
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
    setVisibleNewsCount(prev => Math.min(prev + 3, regularNews.length));
    setTimeout(() => {
      if (newsContainerRef.current) {
        newsContainerRef.current.scrollIntoView({ behavior: 'smooth', block: 'end' });
      }
    }, 100);
  };
  
  // Function to show less news
  const showLessNews = () => {
    setVisibleNewsCount(prev => Math.max(prev - 3, 9));
  };

  // Function to show explanation modal
  const showExplanation = (news: NewsItem) => {
    setSelectedNews(news);
    setShowModal(true);
  };

  // Function to close modal
  const closeModal = () => {
    setShowModal(false);
    setSelectedNews(null);
  };

  // Function to toggle dark mode
  const toggleDarkMode = () => {
    setIsDarkMode(!isDarkMode);
  };
  
  // Create rows of news items
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

  // Define theme classes
  const themeClasses = {
    background: isDarkMode 
      ? "bg-gradient-to-br from-gray-900 via-slate-900 to-purple-900" 
      : "bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100",
    cardBg: isDarkMode 
      ? "bg-gray-800/90 border-gray-700/50" 
      : "bg-white/80 border-white/50",
    cardBgSecondary: isDarkMode 
      ? "bg-gray-800/80 border-gray-700/50" 
      : "bg-white/80 border-white/50",
    text: isDarkMode ? "text-gray-100" : "text-gray-800",
    textSecondary: isDarkMode ? "text-gray-300" : "text-gray-600",
    textMuted: isDarkMode ? "text-gray-400" : "text-gray-500",
    modalBg: isDarkMode ? "bg-gray-800" : "bg-white",
    modalHeaderBg: isDarkMode 
      ? "bg-gradient-to-r from-purple-700 via-blue-700 to-purple-800" 
      : "bg-gradient-to-r from-blue-600 via-purple-600 to-blue-800",
    modalFooterBg: isDarkMode 
      ? "bg-gradient-to-r from-gray-800 to-gray-700 border-gray-700" 
      : "bg-gradient-to-r from-gray-50 to-blue-50 border-gray-200",
    tagBg: isDarkMode ? "bg-gray-700 text-gray-200" : "bg-gray-100 text-gray-600",
    linkHover: isDarkMode ? "hover:bg-gray-700" : "hover:bg-blue-100",
    progressBg: isDarkMode ? "bg-gray-700" : "bg-gray-200"
  };
  
  return (
    <div className={`min-h-screen ${themeClasses.background} w-full flex flex-col items-center relative overflow-hidden transition-all duration-500`}>
      {/* Dark Mode Toggle Button */}
      <button
        onClick={toggleDarkMode}
        className={`fixed top-6 right-6 z-20 ${isDarkMode ? 'bg-yellow-500 hover:bg-yellow-400' : 'bg-indigo-600 hover:bg-indigo-700'} text-white rounded-full w-14 h-14 flex items-center justify-center shadow-lg hover:shadow-xl transition-all duration-300 hover:scale-110 backdrop-blur-sm border-2 ${isDarkMode ? 'border-yellow-400/30' : 'border-indigo-500/30'}`}
      >
        {isDarkMode ? (
          <Sun className="w-6 h-6 animate-pulse" />
        ) : (
          <Moon className="w-6 h-6" />
        )}
      </button>

      {/* Animated background elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className={`absolute -top-40 -right-40 w-80 h-80 ${isDarkMode ? 'bg-purple-600' : 'bg-blue-400'} rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-pulse`}></div>
        <div className={`absolute -bottom-40 -left-40 w-80 h-80 ${isDarkMode ? 'bg-blue-600' : 'bg-purple-400'} rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-pulse`}></div>
        <div className={`absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-60 h-60 ${isDarkMode ? 'bg-indigo-500' : 'bg-pink-300'} rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-pulse`}></div>
      </div>
      
      <div className="w-full max-w-6xl px-6 relative z-10">
        {/* Hero Header */}
        <div className="text-center py-8 mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-r from-blue-600 to-purple-600 rounded-2xl mb-4 shadow-lg">
            <Info className="w-8 h-8 text-white" />
          </div>
          <h1 className={`text-4xl font-bold ${isDarkMode ? 'bg-gradient-to-r from-gray-100 via-blue-300 to-purple-300' : 'bg-gradient-to-r from-gray-900 via-blue-800 to-purple-800'} bg-clip-text text-transparent mb-2 transition-all duration-500`}>
            FakeNews-Verifier
          </h1>
          <p className={`${themeClasses.textSecondary} text-lg transition-colors duration-500`}>Artificial intelligence news verification</p>
        </div>

        <h2 className={`${themeClasses.text} text-2xl font-bold text-center mb-8 flex items-center justify-center transition-colors duration-500`}>
          <span className="bg-gradient-to-r from-blue-600 to-purple-600 w-1 h-8 rounded-full mr-3"></span>
          Featured News
          <span className="bg-gradient-to-r from-purple-600 to-blue-600 w-1 h-8 rounded-full ml-3"></span>
        </h2>
        
        <div className="relative flex justify-center mb-4">
          {/* Left arrow outside carousel */}
          <button 
            onClick={prevSlide}
            className="absolute left-0 top-1/2 transform -translate-y-1/2 -translate-x-12 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white rounded-full w-12 h-12 flex items-center justify-center z-10 shadow-lg transition-all duration-300 hover:scale-110 hover:shadow-xl">
           <ChevronLeft className="w-5 h-5"/>
          </button>
          
          <div className="w-full">
            <div className="flex relative">
              {/* Left (smaller) slide - vertically centered with fixed height */}
              <div className="w-1/4 px-2 flex items-center">
                <div className={`${themeClasses.cardBgSecondary} backdrop-blur-sm rounded-2xl overflow-hidden shadow-lg hover:shadow-xl transition-all duration-300 h-64 flex flex-col w-full border hover:scale-105`}>
                  <img src={featuredNews[leftIndex].image} alt={featuredNews[leftIndex].title} className="w-full h-32 object-cover flex-shrink-0" />
                  <div className="p-3 flex-1 flex flex-col justify-between">
                    <div>
                      <h3 className={`text-xs font-semibold line-clamp-2 h-8 overflow-hidden mb-2 ${themeClasses.text} transition-colors duration-500`}>{featuredNews[leftIndex].title}</h3>
                      <div className={`flex justify-between text-xs ${themeClasses.textSecondary} transition-colors duration-500`}>
                        <p className='truncate'>Source: {featuredNews[leftIndex].source}</p>
                        <p>{featuredNews[leftIndex].date}</p>
                      </div>
                    </div>
                    <div className="mt-2">
                      <div className={`${themeClasses.progressBg} rounded-full h-1.5 mb-2 transition-colors duration-500`}>
                        <div 
                          className={`${getVeracityColor(featuredNews[leftIndex].veracity)} h-1.5 rounded-full transition-all duration-700`} 
                          style={{ width: `${featuredNews[leftIndex].veracity}%` }}
                        ></div>
                      </div>
                      <p className="text-xs text-right flex items-center justify-end">
                        <button onClick={() => showExplanation(featuredNews[leftIndex])} className="focus:outline-none mr-1 flex items-center hover:scale-110 transition-transform">
                          <Info className={`w-3 h-3 hover:text-blue-500 cursor-pointer transition-colors ${themeClasses.textMuted}`} />
                        </button>
                        <span className={`${themeClasses.text} font-medium transition-colors duration-500`}>Truthfulness: {featuredNews[leftIndex].veracity}%</span>
                      </p>
                    </div>
                  </div>
                </div>
              </div>
              
              {/* Center (main) slide with fixed height */}
              <div className="w-2/4 px-2">
                <div className={`${themeClasses.cardBg} backdrop-blur-sm rounded-2xl overflow-hidden shadow-xl hover:shadow-2xl transition-all duration-300 h-96 flex flex-col border hover:scale-[1.02]`}>
                  <img src={featuredNews[centerIndex].image} alt={featuredNews[centerIndex].title} className="w-full h-48 object-cover flex-shrink-0" />
                  <div className="p-4 flex-1 flex flex-col justify-between">
                    <div>
                      <div className="flex items-start justify-between mb-3">
                        <h3 className={`text-lg font-bold line-clamp-2 h-14 overflow-hidden flex-1 ${themeClasses.text} transition-colors duration-500`}>{featuredNews[centerIndex].title}</h3>
                        <a href={featuredNews[centerIndex].url} className={`ml-3 flex-shrink-0 ${isDarkMode ? 'bg-gray-700 hover:bg-gray-600' : 'bg-blue-100 hover:bg-blue-200'} p-2 rounded-full transition-colors duration-300`}>
                          <LinkIcon className="w-4 h-4 text-blue-600"/>
                        </a>
                      </div>
                      <div className={`flex justify-between text-sm ${themeClasses.textSecondary} transition-colors duration-500`}>
                        <span className={`${themeClasses.tagBg} px-3 py-1 rounded-full text-xs font-medium transition-colors duration-500`}>{featuredNews[centerIndex].source}</span>
                        <p>{featuredNews[centerIndex].date}</p>
                      </div>
                    </div>
                    <div className="mt-3">
                      <div className={`${themeClasses.progressBg} rounded-full h-2.5 mb-3 transition-colors duration-500`}>
                        <div 
                          className={`${getVeracityColor(featuredNews[centerIndex].veracity)} h-2.5 rounded-full transition-all duration-700`} 
                          style={{ width: `${featuredNews[centerIndex].veracity}%` }}
                        ></div>
                      </div>
                      <p className="text-sm text-right flex items-center justify-end">
                        <button onClick={() => showExplanation(featuredNews[centerIndex])} className="focus:outline-none mr-2 flex items-center hover:scale-110 transition-transform">
                          <Info className={`w-4 h-4 hover:text-blue-500 cursor-pointer transition-colors ${themeClasses.textMuted}`} />
                        </button>
                        <span className={`${themeClasses.text} font-semibold transition-colors duration-500`}>Truthfulness: {featuredNews[centerIndex].veracity}%</span>
                      </p>
                    </div>
                  </div>
                </div>
              </div>
              
              {/* Right (smaller) slide - vertically centered with fixed height */}
              <div className="w-1/4 px-2 flex items-center">
                <div className={`${themeClasses.cardBgSecondary} backdrop-blur-sm rounded-2xl overflow-hidden shadow-lg hover:shadow-xl transition-all duration-300 h-64 flex flex-col w-full border hover:scale-105`}>
                  <img src={featuredNews[rightIndex].image} alt={featuredNews[rightIndex].title} className="w-full h-32 object-cover flex-shrink-0" />
                  <div className="p-3 flex-1 flex flex-col justify-between">
                    <div>
                      <h3 className={`text-xs font-semibold line-clamp-2 h-8 overflow-hidden mb-2 ${themeClasses.text} transition-colors duration-500`}>{featuredNews[rightIndex].title}</h3>
                      <div className={`flex justify-between text-xs ${themeClasses.textSecondary} transition-colors duration-500`}>
                        <p className='truncate'>Source: {featuredNews[rightIndex].source}</p>
                        <p>{featuredNews[rightIndex].date}</p>
                      </div>
                    </div>
                    <div className="mt-2">
                      <div className={`${themeClasses.progressBg} rounded-full h-1.5 mb-2 transition-colors duration-500`}>
                        <div 
                          className={`${getVeracityColor(featuredNews[rightIndex].veracity)} h-1.5 rounded-full transition-all duration-700`} 
                          style={{ width: `${featuredNews[rightIndex].veracity}%` }}
                        ></div>
                      </div>
                      <p className="text-xs text-right flex items-center justify-end">
                        <button onClick={() => showExplanation(featuredNews[rightIndex])} className="focus:outline-none mr-1 flex items-center hover:scale-110 transition-transform">
                          <Info className={`w-3 h-3 hover:text-blue-500 cursor-pointer transition-colors ${themeClasses.textMuted}`} />
                        </button>
                        <span className={`${themeClasses.text} font-medium transition-colors duration-500`}>Truthfulness: {featuredNews[rightIndex].veracity}%</span>
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
          
          {/* Right arrow outside carousel */}
          <button 
            onClick={nextSlide}
            className="absolute right-0 top-1/2 transform -translate-y-1/2 translate-x-12 bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white rounded-full w-12 h-12 flex items-center justify-center z-10 shadow-lg transition-all duration-300 hover:scale-110 hover:shadow-xl">
            <ChevronRight className="w-5 h-5"/>
          </button>
        </div>
        
        <div className="flex justify-center gap-3 mb-8">
          {featuredNews.map((_, index) => (
            <button 
              key={index} 
              onClick={() => goToSlide(index)}
              className={`h-2 rounded-full transition-all duration-300 ${
                currentSlide === index 
                  ? 'w-8 bg-gradient-to-r from-blue-600 to-purple-600 shadow-lg' 
                  : `w-2 ${isDarkMode ? 'bg-gray-600 hover:bg-gray-500' : 'bg-gray-300 hover:bg-gray-400'}`
              }`}
            ></button>
          ))}
        </div>
        
        <div className="w-full flex justify-center mb-12">
          <div className={`w-full max-w-2xl h-px ${isDarkMode ? 'bg-gradient-to-r from-transparent via-gray-600 to-transparent' : 'bg-gradient-to-r from-transparent via-gray-300 to-transparent'} transition-all duration-500`}></div>
        </div>
        
        {/* All News Section */}
        <div className="w-full mb-12" ref={newsContainerRef}>
          <div className="flex justify-between items-center mb-6">
            <h2 className={`${themeClasses.text} text-2xl font-bold flex items-center transition-colors duration-500`}>
              <span className="bg-gradient-to-r from-purple-600 to-blue-600 w-1 h-8 rounded-full mr-3"></span>
              All News
            </h2>
          </div>
          
          <div className="news-grid space-y-6">
            {newsRows.map((row, rowIndex) => (
              <div key={rowIndex} className="flex -mx-2">
                {row.map((news, colIndex) => (
                  news ? (
                    <div key={colIndex} className="w-1/3 px-3">
                      <div className={`${themeClasses.cardBgSecondary} backdrop-blur-sm rounded-2xl overflow-hidden shadow-lg hover:shadow-xl transition-all duration-300 h-full border hover:scale-[1.02] group`}>
                        <img src={news.image} alt={news.title} className="w-full h-32 object-cover group-hover:scale-105 transition-transform duration-300" />
                        <div className="p-4">
                          <div className="flex justify-between items-start mb-3">
                            <h3 className={`text-sm font-semibold line-clamp-2 flex-1 mr-2 ${themeClasses.text} group-hover:text-opacity-90 transition-all duration-300`}>
                              {news.title}
                            </h3>
                            <a href={news.url} className={`hover:text-blue-600 flex-shrink-0 ${isDarkMode ? 'bg-gray-700 hover:bg-gray-600' : 'bg-blue-50 hover:bg-blue-100'} p-2 rounded-full transition-all duration-200`}>
                              <LinkIcon className="w-4 h-4"/>
                            </a>
                          </div>                          
                          <div className="flex justify-between text-xs mt-2 mb-3">
                            <span className={`${themeClasses.tagBg} px-2 py-1 rounded-full text-xs font-medium transition-colors duration-500`}>{news.source}</span>
                            <p className={`${themeClasses.textMuted} transition-colors duration-500`}>{news.date}</p>
                          </div>
                          <div className={`mt-3 ${themeClasses.progressBg} rounded-full h-1.5 transition-colors duration-500`}>
                            <div 
                              className={`${getVeracityColor(news.veracity)} h-1.5 rounded-full transition-all duration-700`} 
                              style={{ width: `${news.veracity}%` }}
                            ></div>
                          </div>
                          <p className="text-xs text-right flex items-center justify-end mt-3">
                            <button onClick={() => showExplanation(news)} className="focus:outline-none mr-2 flex items-center hover:scale-110 transition-transform">
                              <Info className={`w-4 h-4 hover:text-blue-500 cursor-pointer transition-colors ${themeClasses.textMuted}`} />
                            </button>
                            <span className={`${themeClasses.text} font-medium transition-colors duration-500`}>Truthfulness: {news.veracity}%</span>
                          </p>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div key={colIndex} className="w-1/3 px-3"></div>
                  )
                ))}
              </div>
            ))}
          </div>

          <div className="flex flex-col sm:flex-row justify-center items-center mt-8 space-y-4 sm:space-y-0 sm:space-x-6">
            {visibleNewsCount < regularNews.length && (
              <button 
                onClick={showMoreNews}
                className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white rounded-2xl px-8 py-3 text-sm font-semibold min-w-[180px] flex items-center justify-center gap-3 shadow-lg hover:shadow-xl transition-all duration-300 hover:scale-105 backdrop-blur-sm border border-white/20">
                <SquareChevronDown className="w-5 h-5" />
                See more news
              </button>
            )}

            {visibleNewsCount > 9 && (
              <button 
                onClick={showLessNews}
                className="bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white rounded-2xl px-8 py-3 text-sm font-semibold min-w-[180px] flex items-center justify-center gap-3 shadow-lg hover:shadow-xl transition-all duration-300 hover:scale-105 backdrop-blur-sm border border-white/20">
                <SquareChevronUp className="w-5 h-5" />
                See less
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Modern Modal for explanation */}
      {showModal && selectedNews && (
        <div className="fixed inset-0 bg-black bg-opacity-60 backdrop-blur-sm flex items-center justify-center z-50 p-4 transition-all duration-300" onClick={closeModal}>
          <div className={`${themeClasses.modalBg} rounded-2xl max-w-3xl w-full max-h-[85vh] overflow-hidden shadow-2xl transform transition-all duration-300 scale-100 animate-pulse-once`} onClick={(e) => e.stopPropagation()}>
            {/* Header with gradient */}
            <div className={`${themeClasses.modalHeaderBg} px-6 py-5`}>
              <div className="flex justify-between items-center">
                <div className="flex items-center space-x-3">
                  <div className="bg-white bg-opacity-20 rounded-full p-2">
                    <Info className="w-6 h-6 text-white" />
                  </div>
                  <div>
                    <h3 className="text-xl font-bold text-white">Truthfulness analysis</h3>
                    <p className="text-blue-100 text-sm">Detailed explanation</p>
                  </div>
                </div>
                <button 
                  onClick={closeModal}
                  className="text-white hover:text-blue-200 transition-colors duration-200 bg-white bg-opacity-10 hover:bg-opacity-20 rounded-full w-10 h-10 flex items-center justify-center group">
                  <span className="text-xl font-light group-hover:scale-110 transition-transform">×</span>
                </button>
              </div>
            </div>
            
            {/* News info bar */}
            <div className={`${isDarkMode ? 'bg-gradient-to-r from-gray-700 to-gray-600 border-gray-600' : 'bg-gradient-to-r from-gray-50 to-blue-50 border-gray-200'} px-6 py-4 border-b transition-all duration-500`}>
              <div className="flex items-center justify-between">
                <div className="flex-1 min-w-0">
                  <h4 className={`text-sm font-semibold ${themeClasses.text} truncate transition-colors duration-500`}>{selectedNews.title}</h4>
                  <p className={`text-xs ${themeClasses.textSecondary} mt-1 flex items-center transition-colors duration-500`}>
                    <span className={`${isDarkMode ? 'bg-blue-700 text-blue-200' : 'bg-blue-100 text-blue-800'} px-2 py-1 rounded-full text-xs font-medium mr-2 transition-colors duration-500`}>
                      {selectedNews.source}
                    </span>
                    {selectedNews.date}
                  </p>
                </div>
                <div className="ml-4 flex items-center space-x-3">
                  <div className="text-right">
                    <p className={`text-xs ${themeClasses.textMuted} mb-1 transition-colors duration-500`}>Truthfulness</p>
                    <div className={`${themeClasses.progressBg} rounded-full h-2 w-20 transition-colors duration-500`}>
                      <div 
                        className={`${getVeracityColor(selectedNews.veracity)} h-2 rounded-full transition-all duration-1000 ease-out`} 
                        style={{ width: `${selectedNews.veracity}%` }}
                      ></div>
                    </div>
                  </div>
                  <div className={`text-2xl font-bold ${selectedNews.veracity >= 75 ? 'text-green-600' : selectedNews.veracity >= 50 ? 'text-yellow-600' : 'text-red-600'}`}>
                    {selectedNews.veracity}%
                  </div>
                </div>
              </div>
            </div>

            {/* Content */}
            <div className="overflow-y-auto flex-1 p-6" style={{ maxHeight: 'calc(85vh - 200px)' }}>
              <div className="prose prose-gray max-w-none">
                {selectedNews.explanation && selectedNews.explanation.length > 0 ? (
                  <div className="space-y-4">
                    {selectedNews.explanation.map((paragraph, index) => (
                      <div key={index} className="flex items-start space-x-3">
                        <div className={`flex-shrink-0 w-6 h-6 ${isDarkMode ? 'bg-blue-800' : 'bg-blue-100'} rounded-full flex items-center justify-center mt-1 transition-colors duration-500`}>
                          <span className={`${isDarkMode ? 'text-blue-200' : 'text-blue-600'} text-xs font-bold transition-colors duration-500`}>{index + 1}</span>
                        </div>
                        <p className={`${themeClasses.text} leading-relaxed text-sm flex-1 transition-colors duration-500`}>
                          {paragraph}
                        </p>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-12">
                    <div className={`${isDarkMode ? 'bg-gradient-to-br from-gray-700 to-gray-600' : 'bg-gradient-to-br from-gray-100 to-gray-200'} rounded-full w-20 h-20 flex items-center justify-center mx-auto mb-4 transition-all duration-500`}>
                      <Info className={`w-10 h-10 ${themeClasses.textMuted} transition-colors duration-500`} />
                    </div>
                    <p className={`${themeClasses.textMuted} text-lg mb-1 transition-colors duration-500`}>No explanation available</p>
                    <p className={`${themeClasses.textMuted} text-sm transition-colors duration-500`}>There is no detailed analysis for this news item.</p>
                  </div>
                )}
              </div>
            </div>

            {/* Footer */}
            <div className={`${themeClasses.modalFooterBg} px-6 py-4 border-t transition-all duration-500`}>
              <div className="flex justify-between items-center">
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                  <p className={`text-xs ${themeClasses.textSecondary} font-medium transition-colors duration-500`}>
                    Analysis generated by FakeNews-Verifier
                  </p>
                </div>
                <a 
                  href={selectedNews.url} 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="inline-flex items-center space-x-2 bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 hover:scale-105 shadow-lg">
                  <span>See source</span>
                  <LinkIcon className="w-4 h-4" />
                </a>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}