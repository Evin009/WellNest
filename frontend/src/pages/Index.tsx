import Navbar from "@/components/Navbar";
import Hero from "@/components/Hero";
import WhyWellNest from "@/components/WhyWellNest";
import CategoryGrid from "@/components/CategoryGrid";
import Contact from "@/components/Contact";
import Footer from "@/components/Footer";

const Index = () => {
  return (
    <div className="min-h-screen">
      <Navbar />
      <Hero />
      <WhyWellNest />
      <CategoryGrid />
      <Contact />
      <Footer />
    </div>
  );
};

export default Index;
