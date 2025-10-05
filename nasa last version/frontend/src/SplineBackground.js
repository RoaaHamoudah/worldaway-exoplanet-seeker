import Spline from '@splinetool/react-spline';

export default function SplineBackground() {
  return (
    <div className="fixed inset-0 overflow-hidden pointer-events-none z-0">
      <main>
        <Spline
          scene="https://prod.spline.design/1nVzPcAFEhMVNeFJ/scene.splinecode" 
        />
      </main>
    </div>
  );
}
