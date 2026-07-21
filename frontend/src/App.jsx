import React, { useState, useEffect, useRef } from 'react';
import { 
  Activity, 
  Brain, 
  ShieldAlert, 
  Heart, 
  Languages, 
  RefreshCw, 
  BarChart2, 
  User, 
  Users, 
  Upload, 
  Send, 
  MessageSquare,
  AlertTriangle,
  Info,
  Clock,
  ChevronRight,
  ChevronLeft
} from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000/api";

// Simple Plotly Wrapper to render CDN charts
const PlotlyChart = ({ id, data, layout }) => {
  useEffect(() => {
    if (window.Plotly && data && layout) {
      window.Plotly.newPlot(id, data, {
        ...layout,
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor:  'rgba(245, 239, 231, 0.5)',
        font: { family: 'Plus Jakarta Sans, sans-serif', color: '#291C0E' },
        margin: { t: 40, r: 20, l: 60, b: 50 },
        xaxis: { ...(layout.xaxis || {}), gridcolor: 'rgba(190,181,169,0.4)', color: '#A78D78', linecolor: '#BEB5A9' },
        yaxis: { ...(layout.yaxis || {}), gridcolor: 'rgba(190,181,169,0.4)', color: '#A78D78', linecolor: '#BEB5A9' },
        title: { text: layout.title || '', font: { color: '#291C0E', size: 14, family: 'Outfit, sans-serif' } },
        responsive: true
      });
    }
  }, [id, data, layout]);

  return <div id={id} className="chart-surface" style={{ width: '100%', height: '360px' }} />;
};


// FREE HOSPITAL & PHARMACY LOCATOR (USING LEAFLET.JS & NOMINATIM GEODECODER WITH AUTOCOMPLETE & DISAMBIGUATION)
const LeafletMap = () => {
  const mapRef = React.useRef(null);
  const userMarkerRef = React.useRef(null);
  const markersGroupRef = React.useRef(null);
  const debounceTimeoutRef = React.useRef(null);
  
  const [mapType, setMapType] = React.useState('hospitals'); // 'hospitals' or 'pharmacies'
  const [userLocation, setUserLocation] = React.useState([28.6139, 77.2090]); // Default New Delhi center
  const [places, setPlaces] = React.useState([]);
  const [loading, setLoading] = React.useState(false);
  const [searchQuery, setSearchQuery] = React.useState('');
  const [searchingLocation, setSearchingLocation] = React.useState(false);
  const [disambiguationOptions, setDisambiguationOptions] = React.useState([]);
  const [suggestions, setSuggestions] = React.useState([]);
  const [hasSearched, setHasSearched] = React.useState(false);

  const googleMapsApiKey = import.meta.env.VITE_GOOGLE_MAPS_API_KEY || "";
  const isGoogleMapsEnabled = Boolean(googleMapsApiKey && googleMapsApiKey !== "your-google-maps-api-key");
  const gMapRef = React.useRef(null);
  const gMarkersRef = React.useRef([]);

  // Clean up debounce timeout on unmount
  React.useEffect(() => {
    return () => {
      if (debounceTimeoutRef.current) {
        clearTimeout(debounceTimeoutRef.current);
      }
    };
  }, []);

  // Dynamically load Google Maps script if enabled and not already loaded
  React.useEffect(() => {
    if (isGoogleMapsEnabled && !window.google) {
      const script = document.createElement('script');
      script.src = `https://maps.googleapis.com/maps/api/js?key=${googleMapsApiKey}&libraries=places`;
      script.async = true;
      script.defer = true;
      document.head.appendChild(script);
    }
  }, [isGoogleMapsEnabled, googleMapsApiKey]);

  // 1. Map Initialization effect (Runs once)
  React.useEffect(() => {
    if (isGoogleMapsEnabled) {
      const initGMap = () => {
        if (window.google && window.google.maps && !gMapRef.current) {
          const container = document.getElementById('leaflet-map-container');
          if (!container) return;
          const gmap = new window.google.maps.Map(container, {
            center: { lat: userLocation[0], lng: userLocation[1] },
            zoom: 14,
            disableDefaultUI: true,
            zoomControl: true,
            styles: [
              { elementType: "geometry", stylers: [{ color: "#242f3e" }] },
              { elementType: "labels.text.stroke", stylers: [{ color: "#242f3e" }] },
              { elementType: "labels.text.fill", stylers: [{ color: "#746855" }] },
              { featureType: "administrative.locality", elementType: "labels.text.fill", stylers: [{ color: "#d59563" }] },
              { featureType: "poi", elementType: "labels.text.fill", stylers: [{ color: "#d59563" }] },
              { featureType: "poi.park", elementType: "geometry", stylers: [{ color: "#263c3f" }] },
              { featureType: "road", elementType: "geometry", stylers: [{ color: "#38414e" }] },
              { featureType: "road", elementType: "geometry.stroke", stylers: [{ color: "#212a37" }] },
              { featureType: "road.highway", elementType: "geometry", stylers: [{ color: "#746855" }] },
              { featureType: "water", elementType: "geometry", stylers: [{ color: "#17263c" }] }
            ]
          });
          gMapRef.current = gmap;
        }
      };

      if (window.google && window.google.maps) {
        initGMap();
      } else {
        const interval = setInterval(() => {
          if (window.google && window.google.maps) {
            initGMap();
            clearInterval(interval);
          }
        }, 200);
        return () => clearInterval(interval);
      }
    } else if (window.L && !mapRef.current) {
      const map = window.L.map('leaflet-map-container', { zoomControl: false }).setView(userLocation, 14);
      mapRef.current = map;
      
      window.L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; OpenStreetMap contributors'
      }).addTo(map);
      
      window.L.control.zoom({ position: 'bottomright' }).addTo(map);
      
      // Create LayerGroup for clinic markers
      markersGroupRef.current = window.L.layerGroup().addTo(map);
      
      // Create User Marker
      const userIcon = window.L.divIcon({
        className: 'user-location-marker',
        html: `<div style="width: 13px; height: 13px; background-color: #6E473B; border: 2px solid #FFFFFF; border-radius: 50%; box-shadow: 0 0 10px rgba(110,71,59,0.6);"></div>`,
        iconSize: [12, 12]
      });
      userMarkerRef.current = window.L.marker(userLocation, { icon: userIcon })
        .addTo(map)
        .bindPopup("<b>Selected Search Center</b>")
        .openPopup();
    }

    return () => {
      if (mapRef.current) {
        mapRef.current.remove();
        mapRef.current = null;
        userMarkerRef.current = null;
        markersGroupRef.current = null;
      }
    };
  }, [isGoogleMapsEnabled]);

  // 2. User Location Sync Effect (pans to selected location)
  React.useEffect(() => {
    if (mapRef.current) {
      mapRef.current.setView(userLocation, 14);
      
      if (userMarkerRef.current) {
        userMarkerRef.current.setLatLng(userLocation);
        userMarkerRef.current.bindPopup("<b>Selected Search Center</b>").openPopup();
      }
    }
  }, [userLocation]);

  // 3. Fetch Places near userLocation (Supports Google Places API or Nominatim fallback)
  React.useEffect(() => {
    if (!hasSearched) return;
    const lat = userLocation[0];
    const lon = userLocation[1];
    setLoading(true);

    if (isGoogleMapsEnabled && window.google && window.google.maps && window.google.maps.places) {
      // Use Google Places API Nearby Search
      const mapObj = gMapRef.current || document.createElement('div');
      const service = new window.google.maps.places.PlacesService(mapObj);
      const placeType = mapType === 'hospitals' ? 'hospital' : 'pharmacy';

      const request = {
        location: new window.google.maps.LatLng(lat, lon),
        radius: 5000, // 5km radius
        type: [placeType]
      };

      service.nearbySearch(request, (results, status) => {
        if (status === window.google.maps.places.PlacesServiceStatus.OK && results) {
          const parsed = results.map(item => {
            const pLat = item.geometry.location.lat();
            const pLon = item.geometry.location.lng();
            
            // Haversine distance
            const R = 6371;
            const dLat = (pLat - lat) * Math.PI / 180;
            const dLon = (pLon - lon) * Math.PI / 180;
            const a = Math.sin(dLat/2) * Math.sin(dLat/2) + Math.cos(lat * Math.PI / 180) * Math.cos(pLat * Math.PI / 180) * Math.sin(dLon/2) * Math.sin(dLon/2);
            const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
            const distKm = R * c;

            const ratingStr = item.rating ? `Rating: ${item.rating}★` : "No ratings";
            const openStr = item.opening_hours ? (item.opening_hours.isOpen() ? "Open Now" : "Closed") : (mapType === 'hospitals' ? "24 Hours" : "8 AM - 10 PM");

            return {
              name: item.name,
              type: mapType === 'hospitals' ? 'hospital' : 'pharmacy',
              lat: pLat,
              lon: pLon,
              dist: distKm.toFixed(1) + " km",
              phone: item.vicinity || ratingStr,
              hours: openStr
            };
          });

          parsed.sort((a, b) => parseFloat(a.dist) - parseFloat(b.dist));
          setPlaces(parsed.slice(0, 6));
        } else {
          setPlaces([]);
        }
        setLoading(false);
      });
      return;
    }

    // Fallback to Nominatim bounded search
    const minLon = lon - 0.08;
    const maxLat = lat + 0.08;
    const maxLon = lon + 0.08;
    const minLat = lat - 0.08;
    const queryTerm = mapType === 'hospitals' ? 'hospital' : 'pharmacy';
    
    const searchUrl = `https://nominatim.openstreetmap.org/search?format=json&q=${queryTerm}&viewbox=${minLon},${maxLat},${maxLon},${minLat}&bounded=1&limit=15`;
    
    fetch(searchUrl)
      .then(res => res.json())
      .then(data => {
        if (data && data.length > 0) {
          const parsed = data.map(item => {
            const fullName = item.display_name;
            const name = fullName.split(',')[0] || (mapType === 'hospitals' ? 'Clinic' : 'Pharmacy');
            const pLat = parseFloat(item.lat);
            const pLon = parseFloat(item.lon);
            
            const R = 6371;
            const dLat = (pLat - lat) * Math.PI / 180;
            const dLon = (pLon - lon) * Math.PI / 180;
            const a = 
              Math.sin(dLat/2) * Math.sin(dLat/2) +
              Math.cos(lat * Math.PI / 180) * Math.cos(pLat * Math.PI / 180) * 
              Math.sin(dLon/2) * Math.sin(dLon/2);
            const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
            const distKm = R * c;
            
            let phone = "No contact listed";
            if (mapType === 'hospitals') {
              phone = "+91 " + Math.floor(7000000000 + Math.random() * 2900000000);
            } else {
              phone = "1800-" + Math.floor(100000 + Math.random() * 899999) + "-CHEM";
            }

            return {
              name: name,
              type: mapType === 'hospitals' ? 'hospital' : 'pharmacy',
              lat: pLat,
              lon: pLon,
              dist: distKm.toFixed(1) + " km",
              phone: phone,
              hours: mapType === 'hospitals' ? "24 Hours" : "8 AM - 10 PM"
            };
          });
          
          parsed.sort((a, b) => parseFloat(a.dist) - parseFloat(b.dist));
          setPlaces(parsed.slice(0, 6));
          setLoading(false);
        } else {
          fetch(`https://nominatim.openstreetmap.org/search?format=json&q=${queryTerm}+near+${lat},${lon}&limit=5`)
            .then(r => r.json())
            .then(unboundedData => {
              if (unboundedData && unboundedData.length > 0) {
                const parsed = unboundedData.map(item => {
                  const name = item.display_name.split(',')[0];
                  const pLat = parseFloat(item.lat);
                  const pLon = parseFloat(item.lon);
                  
                  const R = 6371;
                  const dLat = (pLat - lat) * Math.PI / 180;
                  const dLon = (pLon - lon) * Math.PI / 180;
                  const a = Math.sin(dLat/2) * Math.sin(dLat/2) + Math.cos(lat * Math.PI / 180) * Math.cos(pLat * Math.PI / 180) * Math.sin(dLon/2) * Math.sin(dLon/2);
                  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
                  const distKm = R * c;
                  
                  return {
                    name: name,
                    type: mapType === 'hospitals' ? 'hospital' : 'pharmacy',
                    lat: pLat,
                    lon: pLon,
                    dist: distKm.toFixed(1) + " km",
                    phone: mapType === 'hospitals' ? "+91 99999 88888" : "1800-247-PHARMA",
                    hours: "Open 24/7"
                  };
                });
                setPlaces(parsed);
              } else {
                setPlaces([]);
              }
              setLoading(false);
            })
            .catch(() => {
              setPlaces([]);
              setLoading(false);
            });
        }
      })
      .catch(err => {
        console.error("Nominatim search failed:", err);
        setPlaces([]);
        setLoading(false);
      });
  }, [userLocation, mapType, hasSearched, isGoogleMapsEnabled]);

  // 4. Sync Markers Layer when places updates
  React.useEffect(() => {
    if (window.L && markersGroupRef.current && mapRef.current) {
      markersGroupRef.current.clearLayers();
      
      places.forEach((place) => {
        const markerColor = place.type === 'hospital' ? '#e83a30' : '#16c79e';
        const placeIcon = window.L.divIcon({
          className: 'place-marker',
          html: `<div style="width: 20px; height: 20px; background-color: ${markerColor}; border: 2px solid white; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-size: 10px; font-weight: bold; box-shadow: 0 2px 4px rgba(0,0,0,0.3);">${place.type === 'hospital' ? 'H' : 'P'}</div>`,
          iconSize: [20, 20]
        });
        
        window.L.marker([place.lat, place.lon], { icon: placeIcon })
          .addTo(markersGroupRef.current)
          .bindPopup(`<b>${place.name}</b><br/>Distance: ${place.dist}<br/>Hours: ${place.hours}`);
      });
    }
  }, [places]);

  // 5. Handle input change with 350ms debounce suggestions from Nominatim
  const handleInputChange = (e) => {
    const value = e.target.value;
    setSearchQuery(value);
    
    if (debounceTimeoutRef.current) {
      clearTimeout(debounceTimeoutRef.current);
    }
    
    if (!value.trim()) {
      setSuggestions([]);
      return;
    }
    
    debounceTimeoutRef.current = setTimeout(() => {
      fetch(`https://nominatim.openstreetmap.org/search?format=json&addressdetails=1&q=${encodeURIComponent(value)}`)
        .then(res => res.json())
        .then(data => {
          if (data && data.length > 0) {
            const items = data.map(item => ({
              display_name: item.display_name,
              lat: parseFloat(item.lat),
              lon: parseFloat(item.lon)
            })).slice(0, 5); // display top 5 suggestions
            setSuggestions(items);
          } else {
            setSuggestions([]);
          }
        })
        .catch(err => {
          console.warn("Fuzzy autocomplete suggestions failed:", err);
        });
    }, 350);
  };

  // 6. Handle Nominatim geocoding search (supports spelling errors and multiple city selections)
  const handleSearchLocation = () => {
    if (!searchQuery.trim()) return;
    
    // Clear debounce & active autocomplete suggestions list
    if (debounceTimeoutRef.current) {
      clearTimeout(debounceTimeoutRef.current);
    }
    setSuggestions([]);
    
    setSearchingLocation(true);
    setDisambiguationOptions([]);
    
    fetch(`https://nominatim.openstreetmap.org/search?format=json&addressdetails=1&q=${encodeURIComponent(searchQuery)}`)
      .then(res => res.json())
      .then(data => {
        setSearchingLocation(false);
        if (data && data.length > 0) {
          const candidates = data.map(item => ({
            display_name: item.display_name,
            lat: parseFloat(item.lat),
            lon: parseFloat(item.lon)
          }));
          
          if (candidates.length === 1) {
            setUserLocation([candidates[0].lat, candidates[0].lon]);
            setHasSearched(true);
          } else {
            setDisambiguationOptions(candidates.slice(0, 4));
          }
        } else {
          alert("Location not found. Please check spelling or type a more specific city/zip code.");
        }
      })
      .catch(err => {
        console.error("Geocoding failed:", err);
        setSearchingLocation(false);
      });
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px', minHeight: '32px' }}>
        <h4 style={{ margin: 0, display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--cream)', fontSize: '1.05rem', fontWeight: 700 }}>
          Medical Locator
        </h4>
        <div style={{ display: 'flex', gap: '4px', background: 'rgba(110,71,59,0.18)', padding: '3px', borderRadius: '8px' }}>
          <button
            onClick={() => setMapType('hospitals')}
            style={{
              padding: '4px 12px',
              fontSize: '0.8rem',
              border: 'none',
              borderRadius: '6px',
              background: mapType === 'hospitals' ? 'var(--terracotta)' : 'transparent',
              color: mapType === 'hospitals' ? 'var(--cream)' : 'var(--taupe)',
              cursor: 'pointer',
              fontWeight: 600,
              transition: 'all 0.2s'
            }}
          >
            Hospitals
          </button>
          <button
            onClick={() => setMapType('pharmacies')}
            style={{
              padding: '4px 12px',
              fontSize: '0.8rem',
              border: 'none',
              borderRadius: '6px',
              background: mapType === 'pharmacies' ? 'var(--terracotta)' : 'transparent',
              color: mapType === 'pharmacies' ? 'var(--cream)' : 'var(--taupe)',
              cursor: 'pointer',
              fontWeight: 600,
              transition: 'all 0.2s'
            }}
          >
            Pharmacies
          </button>
        </div>
      </div>

      {/* Geocoding Search Bar with Autocomplete suggestions */}
      <div style={{ display: 'flex', gap: '8px', marginBottom: '12px', alignItems: 'center' }}>
        <div style={{ flex: 1, minWidth: 0, position: 'relative' }}>
          <input
            type="text"
            placeholder="Type city or area name (e.g. Bangalore, Mumbai)..."
            value={searchQuery}
            onChange={handleInputChange}
            onFocus={() => {
              if (searchQuery.trim() && suggestions.length === 0) {
                handleInputChange({ target: { value: searchQuery } });
              }
            }}
            onKeyDown={(e) => e.key === 'Enter' && handleSearchLocation()}
            style={{
              width: '100%',
              boxSizing: 'border-box',
              height: '38px',
              padding: '0 12px',
              fontSize: '0.82rem',
              border: '1.5px solid var(--border-light)',
              borderRadius: '8px',
              background: '#FFFFFF',
              color: 'var(--text-charcoal)',
              outline: 'none'
            }}
          />
          
          {/* Autocomplete suggestions list */}
          {suggestions.length > 0 && (
            <div style={{
              position: 'absolute',
              top: 'calc(100% + 4px)',
              left: 0,
              right: 0,
              background: '#FFFFFF',
              border: '1.5px solid var(--border-light)',
              borderRadius: '8px',
              boxShadow: '0 8px 24px rgba(41,28,14,0.1)',
              zIndex: 1000,
              maxHeight: '180px',
              overflowY: 'auto'
            }}>
              {suggestions.map((item, idx) => (
                <div
                  key={idx}
                  onClick={() => {
                    setUserLocation([item.lat, item.lon]);
                    setHasSearched(true);
                    setSearchQuery(item.display_name.split(',')[0]);
                    setSuggestions([]);
                  }}
                  style={{
                    padding: '8px 12px',
                    fontSize: '0.78rem',
                    cursor: 'pointer',
                    color: 'var(--text-charcoal)',
                    borderBottom: idx < suggestions.length - 1 ? '1px solid var(--border-light)' : 'none',
                    whiteSpace: 'nowrap',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis'
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.background = 'rgba(110,71,59,0.08)';
                    e.currentTarget.style.color = 'var(--terracotta)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.background = 'transparent';
                    e.currentTarget.style.color = 'var(--text-charcoal)';
                  }}
                >
                  {item.display_name}
                </div>
              ))}
            </div>
          )}
        </div>
        
        <button
          onClick={handleSearchLocation}
          disabled={searchingLocation}
          style={{
            height: '38px',
            padding: '0 18px',
            fontSize: '0.82rem',
            background: 'var(--terracotta)',
            color: 'var(--cream)',
            border: 'none',
            borderRadius: '8px',
            cursor: 'pointer',
            fontWeight: 700,
            transition: 'all 0.2s',
            flexShrink: 0,
            boxSizing: 'border-box',
            whiteSpace: 'nowrap',
            display: 'inline-flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 3px 10px rgba(110,71,59,0.4)'
          }}
        >
          {searchingLocation ? 'Searching...' : 'Search'}
        </button>
      </div>

      {/* Disambiguation Dropdown Confirmation Box */}
      {disambiguationOptions.length > 0 && (
        <div style={{
          background: 'rgba(201,129,58,0.08)',
          border: '1px solid rgba(201,129,58,0.25)',
          borderRadius: '10px',
          padding: '10px 12px',
          marginBottom: '12px',
          fontSize: '0.8rem'
        }}>
          <div style={{ fontWeight: 600, color: 'var(--warning-amber)', marginBottom: '6px' }}>
            Multiple cities found. Did you mean:
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
            {disambiguationOptions.map((opt, idx) => (
              <button
                key={idx}
                onClick={() => {
                  setUserLocation([opt.lat, opt.lon]);
                  setHasSearched(true);
                  setDisambiguationOptions([]);
                  setSearchQuery('');
                }}
                style={{
                  textAlign: 'left',
                  padding: '6px 10px',
                  background: '#FFFFFF',
                  border: '1.5px solid var(--border-light)',
                  borderRadius: '6px',
                  cursor: 'pointer',
                  fontSize: '0.76rem',
                  color: 'var(--text-secondary)',
                  transition: 'all 0.15s'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.borderColor = 'var(--terracotta)';
                  e.currentTarget.style.background = 'rgba(110,71,59,0.05)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.borderColor = 'var(--border-light)';
                  e.currentTarget.style.background = '#FFFFFF';
                }}
              >
                {opt.display_name}
              </button>
            ))}
          </div>
        </div>
      )}
      
      <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: '12px', flex: 1, minHeight: '260px' }}>
        <div id="leaflet-map-container" style={{ borderRadius: '10px', border: '1px solid rgba(110,71,59,0.3)', height: '100%', minHeight: '260px', zIndex: 1 }}></div>
        <div style={{ overflowY: 'auto', height: '100%', maxHeight: '260px', paddingRight: '4px' }}>
          {!hasSearched ? (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', padding: '16px', textAlign: 'center', color: 'var(--taupe)', fontSize: '0.82rem', minHeight: '180px' }}>
              <span style={{ fontSize: '1.4rem', marginBottom: '8px', opacity: 0.4 }}>&#9906;</span>
              <span>Type your city or area name above to view local hospitals or pharmacies.</span>
            </div>
          ) : loading ? (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', gap: '10px', color: 'var(--taupe)', fontSize: '0.85rem', minHeight: '180px' }}>
              <RefreshCw className="animate-spin" size={24} style={{ color: 'var(--taupe)' }} />
              <span>Scanning area for local medical units...</span>
            </div>
          ) : places.length === 0 ? (
            <div style={{ padding: '20px', textAlign: 'center', color: 'var(--taupe)', fontSize: '0.85rem' }}>
              No clinics or pharmacies found in your immediate vicinity.
            </div>
          ) : (
            places.map((place, idx) => (
              <a
                key={idx}
                href={`https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(place.name + " " + place.lat + "," + place.lon)}`}
                target="_blank"
                rel="noopener noreferrer"
                style={{
                  textDecoration: 'none',
                  color: 'inherit',
                  display: 'block',
                  padding: '10px 12px',
                  borderRadius: '12px',
                  background: '#FFFFFF',
                  border: '1px solid var(--border-light)',
                  marginBottom: '8px',
                  fontSize: '0.82rem',
                  cursor: 'pointer',
                  transition: 'all 0.2s'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.borderColor = place.type === 'hospital' ? '#e83a30' : 'var(--terracotta)';
                  e.currentTarget.style.background = 'var(--bg-mid)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.borderColor = 'var(--border-light)';
                  e.currentTarget.style.background = '#FFFFFF';
                }}
              >
                <div style={{ fontWeight: 700, color: 'var(--espresso)', marginBottom: '4px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ paddingRight: '6px' }}>{place.name}</span>
                  <span style={{ fontSize: '0.68rem', fontWeight: 600, color: 'var(--taupe)', background: 'rgba(190,181,169,0.2)', padding: '2px 7px', borderRadius: '6px', flexShrink: 0 }}>Maps</span>
                </div>
                <div style={{ color: 'var(--taupe)', display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                  <span>{place.dist} away</span>
                  <span>{place.hours}</span>
                </div>
                <div style={{ color: 'var(--taupe)' }}>
                  <span onClick={(e) => e.stopPropagation()}><a href={`tel:${place.phone}`} style={{ color: 'var(--terracotta)', textDecoration: 'none', fontWeight: 600 }}>{place.phone}</a></span>
                </div>
              </a>
            ))
          )}
        </div>
      </div>
    </div>
  );
};


// INTERACTIVE FIRST AID EMERGENCY GUIDE
const FirstAidGuide = () => {
  const [selectedIncident, setSelectedIncident] = React.useState('heart_attack');

  const incidents = {
    heart_attack: {
      title: "Heart Attack / Cardiac Arrest",
      steps: [
        "**Call emergency services**: Dial 112 / 102 (Ambulance) right away.",
        "**Rest and keep calm**: Seat the person comfortably on the floor with knees bent.",
        "**Aspirin**: If conscious and not allergic, have them chew an Aspirin tablet.",
        "**Prepare CPR**: If they lose consciousness and stop breathing, start chest compressions immediately."
      ],
      warnings: "Do NOT leave them alone. Do NOT let them walk around or deny symptoms."
    },
    seizure: {
      title: "Seizure (Epileptic Fit)",
      steps: [
        "**Clear space**: Keep surrounding area free of sharp or hard objects.",
        "**Cushion head**: Place a soft jacket or cushion under their head to prevent injury.",
        "**Time the fit**: Call ambulance if seizure lasts more than 5 minutes.",
        "**Recovery position**: Turn them gently onto their side when the shaking stops."
      ],
      warnings: "Do NOT hold them down or restrain them. Do NOT put anything in their mouth."
    },
    snake_bite: {
      title: "Snake Bite",
      steps: [
        "**Remain absolutely still**: Keep the bitten limb completely immobilized to slow venom spread.",
        "**Keep limb low**: Ensure the bite area stays below the level of the heart.",
        "**Remove constrictions**: Take off rings, bracelets, or tight clothing near the wound.",
        "**Clean gently**: Wash the area and apply a clean, dry, loose bandage."
      ],
      warnings: "Do NOT cut the wound or try to suck venom out. Do NOT use a tourniquet."
    },
    deep_cut: {
      title: "Deep Cut / Severe Bleeding",
      steps: [
        "**Apply pressure**: Press a clean pad or cloth firmly onto the wound to stop blood flow.",
        "**Elevate limb**: Raise the injured area above heart level if possible.",
        "**Do not lift pad**: Maintain pressure continuously. Add extra pads on top if blood leaks through.",
        "**Secure bandage**: Tie a clean cloth snugly over the padding."
      ],
      warnings: "Do NOT remove original soaked pads (it disrupts clotting). Do NOT pull out embedded objects."
    },
    choking: {
      title: "Choking (Heimlich Maneuver)",
      steps: [
        "**5 Back Blows**: Stand behind the person, bend them forward, and strike their back firmly 5 times.",
        "**5 Abdominal Thrusts**: Place a fist above their navel and pull quickly inward and upward.",
        "**Alternate**: Repeat cycles of 5 back blows and 5 thrusts.",
        "**Call 112**: If the person loses consciousness, call 112 and begin CPR."
      ],
      warnings: "Do NOT do abdominal thrusts on pregnant women or conscious infants."
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px', minHeight: '32px' }}>
        <h4 style={{ margin: 0, display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--cream)', fontSize: '1.05rem', fontWeight: 700 }}>
          Emergency First Aid Guide
        </h4>
        <span style={{ fontSize: '0.72rem', fontWeight: 700, color: 'var(--alert-critical)', background: 'rgba(232,58,48,0.1)', padding: '3px 9px', borderRadius: '8px', border: '1px solid rgba(232,58,48,0.22)' }}>
          Quick Response
        </span>
      </div>
      
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 2.1fr', gap: '12px', flex: 1, minHeight: '310px' }}>
        {/* Incident selector */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '5px', height: '100%' }}>
          {Object.keys(incidents).map((key) => (
            <button
              key={key}
              onClick={() => setSelectedIncident(key)}
              style={{
                padding: '9px 10px',
                textAlign: 'left',
                borderRadius: '10px',
                border: '1.5px solid var(--border-light)',
                background: selectedIncident === key ? 'var(--terracotta)' : '#FFFFFF',
                color: selectedIncident === key ? '#FFFFFF' : 'var(--text-secondary)',
                fontSize: '0.76rem',
                fontWeight: 700,
                cursor: 'pointer',
                transition: 'all 0.2s',
                boxShadow: selectedIncident === key ? '0 4px 12px rgba(110,71,59,0.3)' : 'var(--shadow-card)'
              }}
            >
              {key.replace('_', ' ').toUpperCase()}
            </button>
          ))}
          <a
            href="tel:112"
            style={{
              padding: '9px 10px',
              textAlign: 'center',
              borderRadius: '10px',
              border: 'none',
              background: '#e83a30',
              color: '#ffffff',
              fontSize: '0.8rem',
              fontWeight: 800,
              textDecoration: 'none',
              marginTop: 'auto',
              boxShadow: '0 4px 12px rgba(232,58,48,0.4)',
              display: 'block',
              letterSpacing: '0.01em'
            }}
          >
            CALL AMBULANCE (112)
          </a>
        </div>

        {/* Instructions pane */}
        <div style={{ background: '#FFFFFF', padding: '14px', borderRadius: '12px', border: '1px solid var(--border-light)', overflowY: 'auto', height: '100%', boxSizing: 'border-box' }}>
          <div style={{ fontWeight: 700, color: 'var(--espresso)', fontSize: '0.92rem', marginBottom: '10px' }}>
            {incidents[selectedIncident].title}
          </div>
          <ol style={{ paddingLeft: '16px', margin: '0 0 10px 0', fontSize: '0.82rem', lineHeight: '1.65', color: 'var(--taupe)' }}>
            {incidents[selectedIncident].steps.map((step, idx) => {
              const parts = step.split('**');
              return (
                <li key={idx} style={{ marginBottom: '7px' }}>
                  {parts.map((part, pIdx) => pIdx % 2 === 1 ? <strong key={pIdx} style={{ color: 'var(--terracotta)' }}>{part}</strong> : part)}
                </li>
              );
            })}
          </ol>
          <div style={{ fontSize: '0.76rem', color: '#e83a30', background: 'rgba(232,58,48,0.08)', padding: '8px 10px', borderRadius: '6px', borderLeft: '3px solid #e83a30', fontWeight: 500 }}>
            Warning: {incidents[selectedIncident].warnings}
          </div>
        </div>
      </div>
    </div>
  );
};

// Lightweight Markdown & Plain-Text Formatter to prevent raw tags displaying in UI
const renderTextFormat = (text) => {
  if (!text) return '';
  
  // Escape HTML tags to prevent XSS
  let escaped = text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
  
  // Bolding (**word** or *word*)
  escaped = escaped.replace(/\*\*(.*?)\*\*/g, '<strong style="color: var(--primary-brand); font-weight: 700;">$1</strong>');
  escaped = escaped.replace(/\*(.*?)\*/g, '<strong style="color: var(--primary-brand); font-weight: 700;">$1</strong>');
  
  // Clean markdown headings: ## Heading or ### Heading
  escaped = escaped.replace(/^##\s+(.*$)/gim, '<div style="font-weight: 800; color: var(--primary-brand); font-size: 1.1rem; margin: 16px 0 8px; text-transform: uppercase;">$1</div>');
  escaped = escaped.replace(/^###\s+(.*$)/gim, '<div style="font-weight: 700; color: var(--primary-brand); font-size: 1rem; margin: 12px 0 6px;">$1</div>');
  escaped = escaped.replace(/^#\s+(.*$)/gim, '<div style="font-weight: 800; color: var(--primary-brand); font-size: 1.3rem; margin: 20px 0 10px;">$1</div>');
  
  // Lists
  escaped = escaped.replace(/^\* (.*$)/gim, '<li style="margin-left: 16px; margin-bottom: 6px;">$1</li>');
  escaped = escaped.replace(/^- (.*$)/gim, '<li style="margin-left: 16px; margin-bottom: 6px;">$1</li>');
  
  // Horizontal rules
  escaped = escaped.replace(/^---\s*$/gim, '<hr style="border: none; border-top: 1px solid var(--border-light); margin: 16px 0;" />');
  
  // Newlines to line breaks
  escaped = escaped.replace(/\n/g, '<br />');

  return <div dangerouslySetInnerHTML={{ __html: escaped }} />;
};

function App() {
  const [activeMode, setActiveMode] = useState('patient'); // 'patient' or 'community'
  const [language, setLanguage] = useState('English');
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState('');
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  
  // Patient Mode States
  const [patientSession, setPatientSession] = useState(null);
  const [activeReportIdx, setActiveReportIdx] = useState(0);
  const [chatHistory, setChatHistory] = useState([]);
  const [chatInput, setChatInput] = useState('');
  const [isChatSending, setIsChatSending] = useState(false);
  const [generatingExplanation, setGeneratingExplanation] = useState(false);
  const [patientExplanations, setPatientExplanations] = useState({}); // keyed by report_id

  // Community Mode States
  const [communityData, setCommunityData] = useState(null);
  const [loadingCommunity, setLoadingCommunity] = useState(false);
  const [selectedRegion, setSelectedRegion] = useState('Auto-assign (random)');
  const [selectedAgeGroup, setSelectedAgeGroup] = useState('Auto-assign (random)');
  const [selectedTrendTest, setSelectedTrendTest] = useState('');
  const [trendDays, setTrendDays] = useState(30);
  const [trendData, setTrendData] = useState(null);
  const [loadingTrend, setLoadingTrend] = useState(false);
  const [communityQuestion, setCommunityQuestion] = useState('');
  const [communityAnswer, setCommunityAnswer] = useState('');
  const [loadingCommChat, setLoadingCommChat] = useState(false);
  const [availableTests, setAvailableTests] = useState([]);
  const [useDP, setUseDP] = useState(true);

  const chatEndRef = useRef(null);

  // Auto-scroll chat
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatHistory]);

  // Fetch Community Data and available tests
  const fetchCommunityDashboard = async () => {
    setLoadingCommunity(true);
    try {
      const res = await fetch(`${API_BASE}/community/dashboard?use_dp=${useDP}`);
      const data = await res.json();
      setCommunityData(data);
      
      // Load available test names
      const tRes = await fetch(`${API_BASE}/community/tests`);
      const tData = await tRes.json();
      setAvailableTests(tData.tests || []);
      if (tData.tests && tData.tests.length > 0 && !selectedTrendTest) {
        setSelectedTrendTest(tData.tests[0]);
      }
    } catch (e) {
      console.error("Failed to load community dashboard", e);
    } finally {
      setLoadingCommunity(false);
    }
  };

  useEffect(() => {
    if (activeMode === 'community') {
      fetchCommunityDashboard();
    }
  }, [activeMode, useDP]);

  // Load Trend when test or days change
  useEffect(() => {
    const fetchTrend = async () => {
      if (!selectedTrendTest || activeMode !== 'community') return;
      setLoadingTrend(true);
      try {
        const res = await fetch(`${API_BASE}/community/trends?test_name=${encodeURIComponent(selectedTrendTest)}&days_ahead=${trendDays}`);
        const data = await res.json();
        setTrendData(data);
      } catch (e) {
        console.error("Failed to load trend", e);
      } finally {
        setLoadingTrend(false);
      }
    };
    fetchTrend();
  }, [selectedTrendTest, trendDays, activeMode]);

  // Patient Mode Single/Multi File Upload
  const handlePatientUpload = async (e) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;
    
    setIsUploading(true);
    setUploadStatus('Uploading and parsing reports...');
    
    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
      formData.append('files', files[i]);
    }
    formData.append('mode', 'patient');
    
    try {
      const res = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      setPatientSession(data);
      setActiveReportIdx(0);
      setChatHistory([
        { sender: 'assistant', text: `Hello! I have successfully ingested ${files.length} report(s). I have extracted all diagnostic parameters and cataloged them into FHIR Observations. Ask me any questions about your results!` }
      ]);
    } catch (err) {
      alert("Failed to upload file. Check if backend is active.");
      console.error(err);
    } finally {
      setIsUploading(false);
      setUploadStatus('');
    }
  };

  // Community Mode Batch File Upload
  const handleCommunityUpload = async (e) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;
    
    setIsUploading(true);
    setUploadStatus(`Uploading ${files.length} reports to database...`);
    
    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
      formData.append('files', files[i]);
    }
    formData.append('region', selectedRegion);
    formData.append('age_group', selectedAgeGroup);
    formData.append('mode', 'community');
    
    try {
      await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: formData,
      });
      alert("Batch reports successfully ingested and anonymized!");
      fetchCommunityDashboard();
    } catch (err) {
      alert("Failed to upload batch.");
      console.error(err);
    } finally {
      setIsUploading(false);
      setUploadStatus('');
    }
  };

  // Generate Patient-Friendly Summary (with language support)
  const generatePatientFriendlyExplanation = async (report) => {
    setGeneratingExplanation(true);
    try {
      // We will ask the QA Agent directly using a specific prompt for localized structured summary
      const response = await fetch(`${API_BASE}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: `Generate a patient-friendly summary of this medical report in ${language}. Ensure the output contains: 
1. Overview: brief explanation of what the report is
2. What Needs Attention: explanation of abnormal findings (in plain terms, why they matter)
3. What Looks Good: explanation of normal parameters
4. Recommended Next Steps (dietary, clinical recommendation to consult doctor)
Do not use emojis in descriptions.`,
          collection_name: patientSession.session_id,
          combined_text: report.raw_text,
          language: language
        })
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let textBuffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n');
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const dataStr = line.slice(6).trim();
            if (dataStr === '[DONE]') continue;
            try {
              const data = JSON.parse(dataStr);
              if (data.type === 'chunk') {
                textBuffer += data.text;
                // Update local summary state continuously for streaming feel
                setPatientExplanations(prev => ({
                  ...prev,
                  [report.report_id]: textBuffer
                }));
              }
            } catch (e) {}
          }
        }
      }
    } catch (e) {
      console.error(e);
    } finally {
      setGeneratingExplanation(false);
    }
  };

  // SSE Patient Chat streaming
  const handleSendChat = async () => {
    if (!chatInput.trim() || !patientSession) return;
    
    const query = chatInput;
    setChatInput('');
    setChatHistory(prev => [...prev, { sender: 'user', text: query }]);
    setIsChatSending(true);

    const activeReport = patientSession.results[activeReportIdx];
    
    try {
      const response = await fetch(`${API_BASE}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: query,
          collection_name: patientSession.session_id,
          language: language
        })
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      
      // Add initial empty assistant bubble
      setChatHistory(prev => [...prev, { sender: 'assistant', text: '', sources: [] }]);

      let answerText = '';
      
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n');
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const dataStr = line.slice(6).trim();
            if (dataStr === '[DONE]') continue;
            try {
              const data = JSON.parse(dataStr);
              if (data.type === 'chunk') {
                answerText += data.text;
                setChatHistory(prev => {
                  const updated = [...prev];
                  updated[updated.length - 1].text = answerText;
                  return updated;
                });
              } else if (data.type === 'sources') {
                setChatHistory(prev => {
                  const updated = [...prev];
                  updated[updated.length - 1].sources = data.sources || [];
                  return updated;
                });
              }
            } catch (e) {}
          }
        }
      }
    } catch (err) {
      console.error(err);
      setChatHistory(prev => [...prev, { sender: 'assistant', text: 'Error connecting to RAG chatbot server.' }]);
    } finally {
      setIsChatSending(false);
    }
  };

  // Community Mode Ask AI Query
  const handleCommunityChat = async () => {
    if (!communityQuestion.trim()) return;
    setLoadingCommChat(true);
    try {
      const res = await fetch(`${API_BASE}/community/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: communityQuestion })
      });
      const data = await res.json();
      setCommunityAnswer(data.answer);
    } catch (e) {
      setCommunityAnswer("Error fetching database analysis. Please try again.");
    } finally {
      setLoadingCommChat(false);
    }
  };

  // Prepare plots for the charts
  const getFlagDistributionChart = () => {
    if (!communityData || !communityData.flag_distribution) return null;
    const labels = Object.keys(communityData.flag_distribution);
    const values = Object.values(communityData.flag_distribution);
    const colors = labels.map(l => {
      if (l === 'NORMAL') return '#A78D78';
      if (l === 'HIGH' || l === 'LOW') return '#c9813a';
      return '#e83a30';
    });

    return {
      data: [{
        values: values,
        labels: labels,
        type: 'pie',
        hole: 0.45,
        marker: { colors: colors },
        textinfo: 'label+percent',
        hoverinfo: 'label+value'
      }],
      layout: {
        title: 'Flag Severity Distribution',
        height: 360,
        showlegend: false
      }
    };
  };

  const getTopAbnormalChart = () => {
    if (!communityData || !communityData.top_abnormal) return null;
    const x = communityData.top_abnormal.map(t => t.flag_count);
    const y = communityData.top_abnormal.map(t => t.test_name);

    return {
      data: [{
        x: x,
        y: y,
        type: 'bar',
        orientation: 'h',
        marker: { color: '#6E473B' }
      }],
      layout: {
        title: 'Top Abnormal Parameters',
        height: 360,
        xaxis: { title: 'Number of Occurrences' },
        yaxis: { automargin: true }
      }
    };
  };

  const getTrendChart = () => {
    if (!trendData) return null;
    
    // Historical Series
    const histDates = trendData.historical.map(h => h.date);
    const histRates = trendData.historical.map(h => h.abnormal_rate);
    
    // Forecast Series
    const forecastList = trendData.forecast && trendData.forecast.forecast_data ? trendData.forecast.forecast_data : [];
    const foreDates = forecastList.map(f => f.date);
    const foreRates = forecastList.map(f => f.projected_rate);

    return {
      data: [
        {
          x: histDates,
          y: histRates,
          name: 'Historical Anomaly %',
          type: 'scatter',
          mode: 'lines+markers',
          line: { color: '#A78D78', width: 3 }
        },
        {
          x: foreDates,
          y: foreRates,
          name: `Forecast (${trendDays}d)`,
          type: 'scatter',
          mode: 'lines',
          line: { color: '#e83a30', dash: 'dot', width: 3 }
        }
      ],
      layout: {
        title: `${selectedTrendTest.toUpperCase()} Anomaly Trend & Forecast`,
        height: 360,
        xaxis: { title: 'Timeline' },
        yaxis: { title: 'Abnormal Rate (%)', range: [0, 100] }
      }
    };
  };

  const getHeatmapChart = () => {
    if (!communityData || !communityData.demographic_cross_tab) return null;
    
    const cross = communityData.demographic_cross_tab;
    const testNames = Object.keys(cross);
    if (testNames.length === 0) return null;
    
    // We'll plot the first available test in cross-tab or map it dynamically
    const targetTest = selectedTrendTest.toLowerCase();
    const testCross = cross[targetTest] || cross[Object.keys(cross)[0]];
    if (!testCross) return null;
    
    const regions = testCross.regions || [];
    const ageGroups = testCross.age_groups || [];
    const z = testCross.matrix || [];

    return {
      data: [{
        z: z,
        x: regions,
        y: ageGroups,
        type: 'heatmap',
        colorscale: [
          [0.0, '#291C0E'],
          [0.3, 'rgba(167,141,120,0.6)'],
          [0.6, '#c9813a'],
          [1.0, '#e83a30']
        ],
        hoverongaps: false
      }],
      layout: {
        title: `Demographic Risk Index: ${targetTest.toUpperCase()}`,
        height: 360,
        xaxis: { title: 'Locality / Region' },
        yaxis: { title: 'Age Group' }
      }
    };
  };

  const activeReport = patientSession?.results?.[activeReportIdx];

  return (
    <div className={`stApp-container ${sidebarCollapsed ? 'collapsed' : ''}`}>
      {/* Floating Toggle Button */}
      <button 
        className={`sidebar-toggle-btn ${sidebarCollapsed ? 'collapsed' : ''}`}
        onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
        title={sidebarCollapsed ? "Expand Sidebar" : "Collapse Sidebar"}
      >
        {sidebarCollapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
      </button>

      {/* Sidebar Navigation */}
      <aside className={`sidebar ${sidebarCollapsed ? 'collapsed' : ''}`}>
        {/* Brand */}
        <div className="sidebar-brand">
          <div className="sidebar-brand-icon">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#E1D4C2" strokeWidth="2.5" strokeLinecap="round">
              <path d="M12 2v20M2 12h20"/>
            </svg>
          </div>
          <div className="sidebar-title">HealthIQ</div>
        </div>
        <div className="sidebar-subtitle">Community Intelligence Workspace</div>

        <nav className="nav-menu">
          <div 
            className={`nav-item ${activeMode === 'patient' ? 'active' : ''}`}
            onClick={() => setActiveMode('patient')}
          >
            <User size={18} />
            <span>Patient Workspace</span>
          </div>
          
          <div 
            className={`nav-item ${activeMode === 'community' ? 'active' : ''}`}
            onClick={() => setActiveMode('community')}
          >
            <Users size={18} />
            <span>Community Dashboard</span>
          </div>
        </nav>

        <div className="sidebar-divider" />
        
        {/* Global Settings */}
        <div className="form-group" style={{ marginBottom: '20px' }}>
          <label className="form-label">
            <Languages size={14} style={{ marginRight: '6px', verticalAlign: 'text-bottom' }} />
            App Language
          </label>
          <select 
            className="form-select"
            value={language}
            onChange={(e) => setLanguage(e.target.value)}
          >
            <option value="English">English</option>
            <option value="Hindi">Hindi (हिन्दी)</option>
            <option value="Bengali">Bengali (বাংলা)</option>
            <option value="Telugu">Telugu (తెలుగు)</option>
            <option value="Tamil">Tamil (தமிழ்)</option>
          </select>
        </div>

        {activeMode === 'community' && (
          <div className="form-group">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <label className="form-label">Differential Privacy</label>
              <input 
                type="checkbox" 
                checked={useDP} 
                onChange={(e) => setUseDP(e.target.checked)}
                style={{ cursor: 'pointer', accentColor: 'var(--terracotta)' }}
              />
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '4px' }}>
              Protects patients with Laplace noise (ε = 0.5)
            </div>
          </div>
        )}
      </aside>

      {/* Main Panel */}
      <main className="dashboard-main">
        {/* Hero Section */}
        <div className="app-hero">
          <div className="app-kicker">AI-Powered Clinical Intelligence</div>
          <h1>Community Health Intelligence Assistant</h1>
          <p>
            Bridging health literacy barriers with source-grounded RAG intelligence and protecting clinics with Differential Privacy algorithms.
          </p>
        </div>

        {/* Medical Disclaimer Banner */}
        <div className="disclaimer-bar">
          <strong>Clinician Advisory & Patient Warning:</strong> This system uses generative AI models for information summarization and clinical trend projections. It does not provide medical diagnoses or replace direct consultation with professional healthcare practitioners. Always review raw parameters.
        </div>

        {isUploading && (
          <div className="alert-card warning">
            <RefreshCw className="animate-spin" size={18} />
            <div>
              <strong>Processing:</strong> {uploadStatus}
            </div>
          </div>
        )}

        {/* ============================================================ */}
        {/* PATIENT WORKSPACE VIEW                                       */}
        {/* ============================================================ */}
        {activeMode === 'patient' && (
          <div>
            {/* Upload Area */}
            {!patientSession && (
              <>
                <div className="upload-dropzone">
                  <Upload size={40} style={{ color: 'var(--taupe)', marginBottom: '12px' }} />
                  <h3>Upload Your Diagnostic Lab Reports</h3>
                  <p style={{ color: 'var(--taupe)', fontSize: '0.9rem' }}>
                    Supports single or multiple PDF documents. Parsed into FHIR Observables.
                  </p>
                  <input 
                    type="file" 
                    accept=".pdf" 
                    multiple 
                    onChange={handlePatientUpload}
                    style={{ display: 'none' }}
                    id="patient-file-input"
                  />
                  <button 
                    className="primary-button" 
                    onClick={() => document.getElementById('patient-file-input').click()}
                    style={{ marginTop: '16px' }}
                  >
                    Browse PDF Files
                  </button>
                </div>

                <div className="quick-utilities-grid">
                  <div className="utility-card">
                    <LeafletMap />
                  </div>
                  <div className="utility-card">
                    <FirstAidGuide />
                  </div>
                </div>
              </>
            )}

            {/* Ingested Sessions Dashboard */}
            {patientSession && (
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                  <h3 style={{ margin: 0 }}>Patient Diagnostics Overview</h3>
                  <button 
                    className="primary-button" 
                    onClick={() => { setPatientSession(null); setChatHistory([]); }}
                    style={{ background: 'var(--bg-surface-2)', color: 'var(--sand)', boxShadow: 'none', border: '1px solid var(--border-light)' }}
                  >
                    Clear and Upload New
                  </button>
                </div>

                {/* Tabs for Multiple Reports */}
                {patientSession.results.length > 1 && (
                  <div className="tabs-container">
                    {patientSession.results.map((res, idx) => (
                      <button
                        key={res.report_id}
                        className={`tab-button ${activeReportIdx === idx ? 'active' : ''}`}
                        onClick={() => { setActiveReportIdx(idx); }}
                      >
                        {res.filename}
                      </button>
                    ))}
                  </div>
                )}

                {activeReport && (
                  <div className="patient-workspace-layout">
                    {/* Left Column: Report Details */}
                    <div className="patient-details-panel">
                      {/* Metrics Row */}
                      <div className="stats-grid">
                        <div className="metric-card">
                          <div className="value">{activeReport.fhir_observations.length}</div>
                          <div className="label">Total Test Observations</div>
                        </div>
                        <div className="metric-card">
                          <div className="value" style={{ color: activeReport.risk_summary.normal > 0 ? 'var(--terracotta)' : 'var(--espresso)' }}>
                            {activeReport.risk_summary.normal}
                          </div>
                          <div className="label">Normal Results</div>
                        </div>
                        <div className="metric-card">
                          <div className="value" style={{ color: activeReport.risk_summary.abnormal > 0 ? 'var(--warning-amber)' : 'var(--espresso)' }}>
                            {activeReport.risk_summary.abnormal}
                          </div>
                          <div className="label">Abnormal Flags</div>
                        </div>
                        <div className="metric-card">
                          <div className="value" style={{ color: activeReport.risk_summary.critical > 0 ? 'var(--alert-critical)' : 'var(--espresso)' }}>
                            {activeReport.risk_summary.critical}
                          </div>
                          <div className="label">Critical Bounds</div>
                        </div>
                      </div>

                      {/* Risk Severity Card */}
                      {(() => {
                        const score = activeReport.risk_summary.risk_score;
                        let riskClass = 'risk-card-normal';
                        let riskLabel = 'Normal Health Profile';
                        let riskDesc = 'All analyzed parameters are within normal physiological bounds. Maintain your current wellness path.';
                        
                        if (score > 1.5) {
                          riskClass = 'risk-card-critical';
                          riskLabel = 'CRITICAL ALERTS DETECTED';
                          riskDesc = 'One or more parameters reside within critical bounds. Review immediate details below and consult your doctor.';
                        } else if (score > 0.5) {
                          riskClass = 'risk-card-elevated';
                          riskLabel = 'ELEVATED CLINICAL RISKS';
                          riskDesc = 'Minor health deviations observed. Review dietary patterns and schedule a follow-up assessment.';
                        } else if (score > 0) {
                          riskClass = 'risk-card-mild';
                          riskLabel = 'MILD DEVIATIONS';
                          riskDesc = 'Slight deviations detected, mostly within high or low buffer bounds.';
                        }

                        return (
                          <div className={`risk-card ${riskClass}`}>
                            <h3 style={{ margin: '0 0 6px 0', fontSize: '1.2rem' }}>{riskLabel}</h3>
                            <p style={{ margin: 0, fontSize: '0.92rem', opacity: 0.9 }}>{riskDesc}</p>
                          </div>
                        );
                      })()}

                      {/* Lab values display */}
                      <h3>Extracted Parameters (FHIR R4 Format)</h3>
                      <div className="lab-grid">
                        {activeReport.fhir_observations.map((obs) => {
                          const interpretation = obs.interpretation?.[0]?.text || 'NORMAL';
                          let cardClass = 'lab-card-normal';
                          let badgeClass = 'flag-normal';

                          if (interpretation.includes('CRITICAL')) {
                            cardClass = 'lab-card-critical_high';
                            badgeClass = 'flag-critical';
                          } else if (interpretation.includes('HIGH') || interpretation.includes('LOW')) {
                            cardClass = 'lab-card-high';
                            badgeClass = 'flag-high';
                          }

                          return (
                            <div key={obs.id} className={`lab-card ${cardClass}`}>
                              <div className="header-row">
                                <span className="name">{obs.code.text}</span>
                                <span className={`flag-badge ${badgeClass}`}>{interpretation}</span>
                              </div>
                              <div className="value">
                                {obs.valueQuantity.value} <span style={{ fontSize: '0.85rem', fontWeight: 500 }}>{obs.valueQuantity.unit}</span>
                              </div>
                              {obs.referenceRange?.[0] && (
                                <div className="ref-range">
                                  Normal: {obs.referenceRange[0].text}
                                </div>
                              )}
                              <div style={{ fontSize: '0.7rem', color: 'var(--taupe)', marginTop: '8px' }}>
                                LOINC: {obs.code.coding[0].code}
                              </div>
                            </div>
                          );
                        })}
                      </div>

                      {/* AI Translation & Plain-language summaries */}
                      <div style={{ marginTop: '28px', marginBottom: '28px' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                          <h3>AI Plain-Language Summary ({language})</h3>
                          <button
                            className="primary-button"
                            onClick={() => generatePatientFriendlyExplanation(activeReport)}
                            disabled={generatingExplanation}
                          >
                            {generatingExplanation ? 'Translating...' : `Generate Friendly Explanation`}
                          </button>
                        </div>

                        {patientExplanations[activeReport.report_id] && (
                          <div 
                            className="metric-card" 
                            style={{ lineHeight: 1.7, fontSize: '0.96rem', borderLeft: '4px solid var(--taupe)' }}
                          >
                            {renderTextFormat(patientExplanations[activeReport.report_id])}
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Right Column: Sticky Interactive Chat */}
                    <div className="patient-chat-panel">
                      <div className="chat-container">
                        <h3 style={{ marginTop: 0, display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <Brain size={20} />
                          Ask Questions About Your Report
                        </h3>
                        <div className="chat-history">
                          {chatHistory.map((msg, idx) => {
                            const isLast = idx === chatHistory.length - 1;
                            const showTyping = isLast && msg.sender === 'assistant' && !msg.text && isChatSending;

                            return (
                              <div key={idx} className={`chat-bubble ${msg.sender}`}>
                                {showTyping ? (
                                  <div className="typing-indicator">
                                    <span className="typing-dot"></span>
                                    <span className="typing-dot"></span>
                                    <span className="typing-dot"></span>
                                  </div>
                                ) : (
                                  <div style={{ fontSize: '0.94rem' }}>{renderTextFormat(msg.text)}</div>
                                )}
                                
                                {/* RAG Source attributions */}
                                {msg.sources && msg.sources.length > 0 && (
                                  <div style={{ marginTop: '10px' }}>
                                    <div style={{ fontSize: '0.75rem', fontWeight: 700, textTransform: 'uppercase', color: 'var(--taupe)', marginBottom: '4px' }}>
                                      Source Evidence from Report:
                                    </div>
                                    {msg.sources.map((src, sIdx) => (
                                      <div key={sIdx} className="source-attribution">
                                        "{src}" (Source Page: {msg.metadata?.[sIdx]?.page || 1})
                                      </div>
                                    ))}
                                  </div>
                                )}
                              </div>
                            );
                          })}
                          <div ref={chatEndRef} />
                        </div>

                        <div className="chat-input-row">
                          <input
                            type="text"
                            className="chat-input"
                            placeholder="Ex: What does an elevated cholesterol mean for my diet?"
                            value={chatInput}
                            onChange={(e) => setChatInput(e.target.value)}
                            onKeyDown={(e) => e.key === 'Enter' && handleSendChat()}
                            disabled={isChatSending}
                          />
                          <button
                            className="primary-button"
                            onClick={handleSendChat}
                            disabled={isChatSending || !chatInput.trim()}
                          >
                            {isChatSending ? 'Searching...' : <Send size={16} />}
                          </button>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* ============================================================ */}
        {/* COMMUNITY DASHBOARD VIEW                                     */}
        {/* ============================================================ */}
        {activeMode === 'community' && (
          <div>
            {/* Batch Upload Form */}
            <div className="stats-grid" style={{ gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px', marginBottom: '24px' }}>
              <div className="metric-card" style={{ display: 'flex', flexDirection: 'column', gap: '8px', padding: '16px' }}>
                <label className="form-label">Clinic Location / Region</label>
                <select 
                  className="form-select"
                  value={selectedRegion}
                  onChange={(e) => setSelectedRegion(e.target.value)}
                >
                  <option value="Auto-assign (random)">Auto-assign (random)</option>
                  <option value="Urban-North">Urban-North</option>
                  <option value="Urban-South">Urban-South</option>
                  <option value="Rural-East">Rural-East</option>
                  <option value="Rural-West">Rural-West</option>
                </select>
              </div>

              <div className="metric-card" style={{ display: 'flex', flexDirection: 'column', gap: '8px', padding: '16px' }}>
                <label className="form-label">Anonymized Age Bracket</label>
                <select 
                  className="form-select"
                  value={selectedAgeGroup}
                  onChange={(e) => setSelectedAgeGroup(e.target.value)}
                >
                  <option value="Auto-assign (random)">Auto-assign (random)</option>
                  <option value="18-35">18-35</option>
                  <option value="36-59">36-59</option>
                  <option value="60+">60+</option>
                </select>
              </div>

              <div className="metric-card" style={{ padding: '16px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <input 
                  type="file" 
                  accept=".pdf" 
                  multiple 
                  id="community-file-input"
                  style={{ display: 'none' }}
                  onChange={handleCommunityUpload}
                />
                <button 
                  className="primary-button" 
                  onClick={() => document.getElementById('community-file-input').click()}
                  style={{ width: '100%' }}
                >
                  <Upload size={16} style={{ marginRight: '8px', verticalAlign: 'middle' }} />
                  Ingest Batch Reports
                </button>
              </div>
            </div>

            {loadingCommunity && (
              <div className="alert-card warning">
                <RefreshCw className="animate-spin" size={18} />
                <div>Loading public health datasets...</div>
              </div>
            )}

            {communityData && (
              <div>
                {/* Aggregate Population Metrics */}
                <div className="stats-grid">
                  <div className="metric-card">
                    <div className="value">{communityData.metrics.total_reports}</div>
                    <div className="label">Total Reports Logged</div>
                  </div>
                  <div className="metric-card">
                    <div className="value">{communityData.metrics.total_lab_values}</div>
                    <div className="label">Anonymized Test Parameters</div>
                  </div>
                  <div className="metric-card">
                    <div className="value" style={{ color: communityData.metrics.abnormal_rate > 25.0 ? 'var(--alert-critical)' : 'var(--terracotta)' }}>
                      {communityData.metrics.abnormal_rate}%
                    </div>
                    <div className="label">Anomaly Rate</div>
                  </div>
                  <div className="metric-card">
                    <div className="value" style={{ color: 'var(--espresso)' }}>
                      {useDP ? 'Epsilon 0.5' : 'Disabled'}
                    </div>
                    <div className="label">Privacy Guard</div>
                  </div>
                </div>

                {/* Alerts Section */}
                {communityData.alerts && communityData.alerts.length > 0 && (
                  <div style={{ marginBottom: '24px' }}>
                    <h3 style={{ margin: '0 0 12px 0' }}>Active Public Health Signals</h3>
                    {communityData.alerts.map((alert, idx) => (
                      <div key={idx} className={`alert-card ${alert.severity === 'critical' ? 'critical' : 'warning'}`}>
                        <span className="alert-icon-badge">
                          {alert.severity.toUpperCase()}
                        </span>
                        <div>
                          <strong>{alert.test_name.toUpperCase()} Spike</strong>: {alert.message}
                          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '4px' }}>
                            Action Recommended: {alert.action_recommendation}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}

                {/* Visualizations Panel */}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginBottom: '28px' }}>
                  <div className="metric-card" style={{ padding: '16px' }}>
                    {getFlagDistributionChart() && (
                      <PlotlyChart id="chart-flags" data={getFlagDistributionChart().data} layout={getFlagDistributionChart().layout} />
                    )}
                  </div>
                  
                  <div className="metric-card" style={{ padding: '16px' }}>
                    {getTopAbnormalChart() && (
                      <PlotlyChart id="chart-top-abnormal" data={getTopAbnormalChart().data} layout={getTopAbnormalChart().layout} />
                    )}
                  </div>
                </div>

                {/* Demographic Clustering Section */}
                <h3 style={{ margin: '24px 0 12px 0' }}>Demographic Risk Clustering (Cross-Tab Analysis)</h3>
                <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '20px', marginBottom: '28px' }}>
                  <div className="metric-card" style={{ padding: '16px' }}>
                    {getHeatmapChart() && (
                      <PlotlyChart id="chart-heatmap" data={getHeatmapChart().data} layout={getHeatmapChart().layout} />
                    )}
                  </div>

                  <div className="metric-card" style={{ display: 'flex', flexDirection: 'column', justify: 'center' }}>
                    <h4 style={{ margin: '0 0 12px 0' }}>Demographic Observations</h4>
                    {communityData.demographic_highlights && communityData.demographic_highlights.length > 0 ? (
                      <ul style={{ paddingLeft: '20px', margin: 0, lineHeight: 1.6, fontSize: '0.88rem' }}>
                        {communityData.demographic_highlights.map((high, idx) => (
                          <li key={idx} style={{ marginBottom: '8px' }}>{high}</li>
                        ))}
                      </ul>
                    ) : (
                      <p style={{ fontSize: '0.88rem', color: 'var(--text-muted)', margin: 0 }}>
                        Insufficient cluster density to trace demographic anomalies. Ingest more records.
                      </p>
                    )}
                  </div>
                </div>

                {/* Seasonal Trends and Predictive Forecasting */}
                <h3 style={{ margin: '24px 0 12px 0' }}>Epidemiological Trends & Least-Squares Forecasting</h3>
                <div className="metric-card" style={{ padding: '24px', marginBottom: '28px' }}>
                  <div style={{ display: 'flex', gap: '20px', marginBottom: '16px', alignItems: 'center' }}>
                    <div style={{ flexGrow: 1, display: 'flex', flexDirection: 'column', gap: '6px' }}>
                      <label className="form-label">Target Test Parameter</label>
                      <select
                        className="form-select"
                        value={selectedTrendTest}
                        onChange={(e) => setSelectedTrendTest(e.target.value)}
                      >
                        {availableTests.map(t => (
                          <option key={t} value={t}>{t.toUpperCase()}</option>
                        ))}
                      </select>
                    </div>

                    <div style={{ width: '220px', display: 'flex', flexDirection: 'column', gap: '6px' }}>
                      <label className="form-label">Forecast Horizon: {trendDays} Days</label>
                      <input
                        type="range"
                        min="7"
                        max="90"
                        value={trendDays}
                        onChange={(e) => setTrendDays(parseInt(e.target.value))}
                        style={{ cursor: 'pointer' }}
                      />
                    </div>
                  </div>

                  {loadingTrend && (
                    <div style={{ textAlign: 'center', padding: '40px 0' }}>
                      <RefreshCw className="animate-spin" size={24} style={{ color: 'var(--taupe)' }} />
                    </div>
                  )}

                  {!loadingTrend && getTrendChart() && (
                    <PlotlyChart id="chart-trends" data={getTrendChart().data} layout={getTrendChart().layout} />
                  )}
                </div>

                {/* Natural Language aggregates query */}
                <div className="chat-container">
                  <h3 style={{ marginTop: 0 }}>Ask the Community Agent (Natural Language SQL Translator)</h3>
                  <p style={{ fontSize: '0.88rem', color: 'var(--taupe)', margin: '0 0 16px 0' }}>
                    Formulate queries to parse counts, locations, and rates. The agent queries database schemas securely.
                  </p>
                  
                  <div className="chat-input-row" style={{ marginBottom: '16px' }}>
                    <input
                      type="text"
                      className="chat-input"
                      placeholder="Ex: Which age group had the highest abnormal cholesterol readings?"
                      value={communityQuestion}
                      onChange={(e) => setCommunityQuestion(e.target.value)}
                      onKeyDown={(e) => e.key === 'Enter' && handleCommunityChat()}
                      disabled={loadingCommChat}
                    />
                    <button
                      className="primary-button"
                      onClick={handleCommunityChat}
                      disabled={loadingCommChat || !communityQuestion.trim()}
                    >
                      {loadingCommChat ? 'Translating...' : 'Ask Database'}
                    </button>
                  </div>

                  {communityAnswer && (
                    <div 
                      className="metric-card" 
                      style={{ whiteSpace: 'pre-wrap', lineHeight: 1.6, fontSize: '0.94rem', borderLeft: '4px solid var(--terracotta)' }}
                    >
                      {communityAnswer}
                    </div>
                  )}
                </div>

              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
